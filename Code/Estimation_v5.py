"""
Concentrated GMM estimation for the replication of
Boucher, Rendall, Ushchev & Zenou (2024) -- consolidated FINAL version v5.

v5 is the single estimation module. It keeps EVERY idea we tried (even the ones
that did not help -- kept as selectable, reportable options) so the whole
project lives in one place.

    python DGP_v5.py
    python Estimation_v5.py      # or:  python run_all_v5.py  (runs everything)

Estimator logic (unchanged since v1):
  1. First stage: predict GPA from OWN covariates + school fixed effects -> yhat.
  2. True peer norm S from actual outcomes; INSTRUMENTS from PREDICTED peer
     outcomes: Shat = S(yhat) and Dhat = dShat/dbeta.
  3. Search the nonlinear parameters (lambda, beta, delta); concentrate gamma.
  4. School-cluster robust standard errors.

================================================================
WHAT CHANGED vs the primitive (Estimation.py), with provenance
================================================================
The primitive did only two-step concentrated GMM + cluster SE. All below is new.

INSTRUMENTS / IDENTIFICATION
  [v5]   Configurable instrument set ACTIVE_INSTRUMENTS. Non-isolated moments =
         own covariates X plus a CHOSEN set of peer-norm transformations.
         Baseline = ["Shat","Dhat"]. Selectable extras so we can SEE which
         estimate worse:
            Dhat2   = (dShat/dbeta)^2   (the "second-derivative" trap)
            Shat2   = Shat^2
            Shat_b1 = predicted norm at beta = 1
            GX      = friends covariates         (distance 1)
            G2X     = friends-of-friends covs    (distance 2, Bramoulle et al.)
         compare_instrument_sets() refits under each and reports beta/SE/F/J.
         What is Dhat?  Dhat = dShat/dbeta: the instrument that IDENTIFIES beta
         (the curvature signal). Without it beta is not identified.
         Friend-of-friend?  The BASELINE does NOT use G2X. The first stage uses
         only OWN covariates, so Shat/Dhat already encode FRIENDS covariates
         (distance 1); G2X is offered here as an explicit alternative.
  [v2-A] Anderson-Rubin weak-IV-robust confidence set for beta (inverts J(beta)).
  [v2]   instrument_strength: partial F and first-stage R^2 for beta.
         (A second-derivative instrument was tested and REMOVED from the baseline
          because the CES norm saturates as beta grows, creating a spurious
          large-beta optimum; it survives as the selectable "Dhat2" so the trap
          stays visible.)

OPTIMIZATION
  [v2-B] Global beta pre-scan to seed the first step (avoids beta~2 / beta~316
         local optima); L-BFGS-B multistart; bounded beta.

ESTIMATORS
  [v3]   CUE (Continuously Updated Estimator): the optimal weight matrix
         W(theta)=Omega(theta)^-1 is RECOMPUTED at every theta inside the
         objective (estimate_model_cue); two-step kept as baseline.
         monte_carlo_cue_vs_two_step compares bias / RMSE / coverage. Finding:
         CUE mainly lightens the tail of beta (fewer blow-ups); the median is
         about the same as two-step.

INFERENCE / TESTS
  [v2-C] Finite-sample cluster correction G/(G-1) in the sandwich SE.
  [v2-F] LIM (beta = 1) restriction test (estimate_lim).
  [v5]   overid_j_test: Hansen J = n*Q over-identification test ("extra moments
         as a test"), reported for two-step and CUE.

STUDIES / POLICY
  [v2-C] identification_frontier: SE(beta) and F vs number of schools.
  [v2-F] planner_marginal_value: key-player targeting v = delta (I-lambda J_S)^-T 1.
  [v2]   monte_carlo_* and robustness_sweep; compute_robust flag skips the heavy
         AR/LIM inside Monte Carlo / frontier loops.

OUTPUT (console)
  [v3/v5] format_full_report: explicit multi-section report (points, cluster SE,
          95% CIs, J-test, instrument strength, Anderson-Rubin, LIM, optimizer).
  [v5]    format_instrument_comparison: the instrument-transformation table.
================================================================

Speed: beta-independent objects are prepared once in prepare_estimation_context;
[v5] G2X / GX / Shat(beta=1) are cached per context (they do not depend on beta).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from DGP_v5 import DATA_DIR, PROJECT_ROOT, X_COLS, load_environment


# (Full tagged change log vs the primitive is in the module docstring above.)
OUTPUT_DIR = PROJECT_ROOT / "outputs_v5"

STARTING_VALUES = [
    np.array([0.10, 2.0, 0.10]),
    np.array([0.25, 8.0, 0.25]),
    np.array([0.50, 15.0, 0.50]),
]

BOUNDS = [
    (0.001, 1),   # lambda
    (0.001, 60.0),   # beta (capped: CES already ~= max well below this)
    (0.001, 1),   # delta
]

REPORT_ROWS = [
    "gamma_age",
    "gamma_female",
    "gamma_f_col",
    "lambda",
    "beta",
    "delta",
    "lambda_1",
    "lambda_2",
    "objective",
]

PARAMETER_NAMES = [
    "gamma_age",
    "gamma_female",
    "gamma_f_col",
    "lambda",
    "beta",
    "delta",
]

DERIVED_PARAMETER_NAMES = [
    "lambda_1",
    "lambda_2",
]


# ============================================================
# 1. First stage and reusable context
# ============================================================

# ============================================================
# [v5] Configurable instrument set for the non-isolated block
# ============================================================
# The non-isolated moments always include own covariates X plus a chosen set of
# peer-norm transformations. "Shat" and "Dhat" are the baseline; the others let
# us TRY alternative instruments and see which identify beta worse.
ACTIVE_INSTRUMENTS = ["Shat", "Dhat"]
_INSTR_NCOLS = {"Shat": 1, "Dhat": 1, "Dhat2": 1, "Shat2": 1,
                "Shat_b1": 1, "GX": len(X_COLS), "G2X": len(X_COLS)}


def _n_extra_cols():  # [v5] number of extra instrument COLUMNS beyond own covariates
    return sum(_INSTR_NCOLS[name] for name in ACTIVE_INSTRUMENTS)


def _extra_instrument_matrix(data):  # [v5] stack the active extra instrument columns
    cols = []
    for name in ACTIVE_INSTRUMENTS:
        v = data[name]
        cols.append(v[:, None] if np.asarray(v).ndim == 1 else v)
    if not cols:
        return np.empty((len(data["X"]), 0))
    return np.concatenate(cols, axis=1)


def _compute_extra_instruments(context, Shat_level, Dhat_level, isolated, demeaning_group):  # [v5]
    """Compute (FE-residualized) candidate instrument columns beyond Shat/Dhat,
    only for the names currently in ACTIVE_INSTRUMENTS."""
    specs = [n for n in ACTIVE_INSTRUMENTS if n not in ("Shat", "Dhat")]
    if not specs:
        return {}
    G = context["G"]; yhat = context["yhat"]; Xlev = context["X_level"]
    def _res(v):
        v = np.asarray(v, float)
        return residualize_fixed_effects(v, estimate_group_fixed_effects(v, demeaning_group))
    out = {}
    if "Dhat2" in specs:
        out["Dhat2"] = _res(Dhat_level ** 2)            # second-derivative-like (the bad idea)
    if "Shat2" in specs:
        out["Shat2"] = _res(Shat_level ** 2)
    cache = context.setdefault("_v5_cache", {})  # [v5] cache beta-independent columns
    if "Shat_b1" in specs:
        if "Shat_b1" not in cache:
            cache["Shat_b1"] = _res(ces_norm(G, yhat, 1.0, isolated))  # predicted norm at beta=1
        out["Shat_b1"] = cache["Shat_b1"]
    if "GX" in specs:
        if "GX" not in cache:
            GX = np.asarray(G @ Xlev)                     # friends' covariates (distance 1)
            cache["GX"] = np.column_stack([_res(GX[:, c]) for c in range(GX.shape[1])])
        out["GX"] = cache["GX"]
    if "G2X" in specs:
        if "G2X" not in cache:
            G2X = np.asarray(G @ (G @ Xlev))              # friends-of-friends covariates (distance 2)
            cache["G2X"] = np.column_stack([_res(G2X[:, c]) for c in range(G2X.shape[1])])
        out["G2X"] = cache["G2X"]
    return out


def first_stage_predicted_outcomes(df):
    """
    Predict GPA from own covariates and school fixed effects.

    The fitted values yhat_i are used only to build peer-based instruments.
    """
    y = df["gpa"].to_numpy(float)
    X = df[X_COLS].to_numpy(float)

    school_dummies = pd.get_dummies(
        df["school_id"],
        prefix="school",
        drop_first=True,
        dtype=float,
    ).to_numpy()

    design = np.column_stack([np.ones(len(df)), X, school_dummies])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ coef

    if np.any(yhat <= 0):
        raise ValueError(
            "Some first-stage predicted outcomes are non-positive. "
            "The CES instrument cannot be constructed. Try increasing the "
            "DGP intercept or reducing the DGP noise."
        )

    return yhat


def estimate_group_fixed_effects(values, group_id):
    """
    Estimate group fixed effects for one vector or matrix.

    Here the groups are school x isolation status. For each observation, this
    returns the mean of its group. Subtracting this object gives the usual
    within-transformed residual.
    """
    values = np.asarray(values, dtype=float)
    one_dimensional = values.ndim == 1

    if one_dimensional:
        values = values[:, None]

    n_groups = int(group_id.max()) + 1
    counts = np.bincount(group_id, minlength=n_groups).astype(float)

    group_sums = np.column_stack(
        [
            np.bincount(group_id, weights=values[:, col], minlength=n_groups)
            for col in range(values.shape[1])
        ]
    )

    group_means = group_sums / counts[:, None]
    fixed_effects = group_means[group_id]

    if one_dimensional:
        return fixed_effects.ravel()

    return fixed_effects


def residualize_fixed_effects(values, fixed_effects):
    """Subtract estimated fixed effects and return residuals."""
    return np.asarray(values, dtype=float) - np.asarray(fixed_effects, dtype=float)


def fixed_effect_residuals(values, group_id):
    """Estimate fixed effects and return residualized values."""
    fixed_effects = estimate_group_fixed_effects(values, group_id)
    return residualize_fixed_effects(values, fixed_effects)


def prepare_estimation_context(df, G_list):
    """
    Prepare all estimation objects that do not vary during optimization.

    This function implements the requested speedups:
    - yhat is computed once.
    - isolated status is computed once.
    - fixed effects for X and y are computed once.
    - residualized X and y are saved once.
    - school networks are combined into one block-diagonal sparse matrix.
    """
    y_level = df["gpa"].to_numpy(float)
    X_level = df[X_COLS].to_numpy(float)
    school_id = df["school_id"].to_numpy()

    yhat = first_stage_predicted_outcomes(df)

    if np.any(y_level <= 0) or np.any(yhat <= 0):
        raise ValueError("CES objects require strictly positive outcomes.")

    G = sparse.block_diag(G_list, format="csr")

    if G.shape[0] != len(df) or G.shape[1] != len(df):
        raise ValueError("Block-diagonal network dimension does not match df.")

    row_sum = np.asarray(G.sum(axis=1)).ravel()
    isolated = row_sum == 0

    school_codes, schools = pd.factorize(school_id, sort=True)
    demeaning_group = 2 * school_codes + isolated.astype(int)

    y_fixed_effect = estimate_group_fixed_effects(y_level, demeaning_group)
    X_fixed_effect = estimate_group_fixed_effects(X_level, demeaning_group)

    y_residual = residualize_fixed_effects(y_level, y_fixed_effect)
    X_residual = residualize_fixed_effects(X_level, X_fixed_effect)

    return {
        "G": G,
        "y": y_residual,
        "X": X_residual,
        "y_level": y_level,
        "X_level": X_level,
        "y_fixed_effect": y_fixed_effect,
        "X_fixed_effect": X_fixed_effect,
        "log_y": np.log(y_level),
        "yhat": yhat,
        "log_yhat": np.log(yhat),
        "school_id": school_id,
        "school_codes": school_codes,
        "schools": schools,
        "demeaning_group": demeaning_group,
        "isolated": isolated,
        "n_isolated": int(np.sum(isolated)),
        "n_nonisolated": int(np.sum(~isolated)),
    }


# ============================================================
# 2. Beta-dependent CES objects
# ============================================================

def ces_norm(G, y, beta, isolated):
    """
    Compute the CES peer norm:

        S_i(beta) = (sum_j g_ij y_j^beta)^(1 / beta)

    Isolated students get zero.
    """
    y = np.asarray(y, dtype=float)

    if np.any(y <= 0):
        raise ValueError("CES norm requires strictly positive outcomes.")

    if beta <= 0:
        raise ValueError("This teaching code assumes beta > 0.")

    y_beta = y ** beta
    A = G @ y_beta

    S = np.zeros_like(y, dtype=float)
    active = ~isolated

    if np.any(A[active] <= 0):
        raise ValueError("The CES norm is undefined for non-positive peer sums.")

    S[active] = A[active] ** (1.0 / beta)

    return S


def ces_norm_derivative(G, y, log_y, beta, isolated, S):
    """
    Compute d S_i(beta) / d beta for the CES peer norm.

    The derivative is:

        dS_i/dbeta = S_i * [
            - log(A_i) / beta^2
            + B_i / (beta A_i)
        ]

    where A_i = sum_j g_ij y_j^beta and
    B_i = sum_j g_ij y_j^beta log(y_j).
    """
    y = np.asarray(y, dtype=float)
    log_y = np.asarray(log_y, dtype=float)

    if np.any(y <= 0):
        raise ValueError("CES derivative requires strictly positive outcomes.")

    if y.shape != log_y.shape:
        raise ValueError("y and log_y must have the same shape.")

    if beta <= 0:
        raise ValueError("beta must be > 0.")

    y_beta = y ** beta

    A = G @ y_beta
    B = G @ (y_beta * log_y)

    D = np.zeros_like(y, dtype=float)
    active = ~isolated

    D[active] = S[active] * (
        -np.log(A[active]) / (beta ** 2)
        + B[active] / (beta * A[active])
    )

    return D


def build_estimation_data(beta, context):
    """
    Construct the beta-dependent estimation arrays.

    X, y, isolated, and fixed-effect groups are already prepared in context.
    Only S, Shat, and Dhat depend on beta.
    """
    isolated = context["isolated"]
    demeaning_group = context["demeaning_group"]

    S_level = ces_norm(
        G=context["G"],
        y=context["y_level"],
        beta=beta,
        isolated=isolated,
    )

    Shat_level = ces_norm(
        G=context["G"],
        y=context["yhat"],
        beta=beta,
        isolated=isolated,
    )

    Dhat_level = ces_norm_derivative(
        G=context["G"],
        y=context["yhat"],
        log_y=context["log_yhat"],
        beta=beta,
        isolated=isolated,
        S=Shat_level,
    )

    S_fixed_effect = estimate_group_fixed_effects(S_level, demeaning_group)
    Shat_fixed_effect = estimate_group_fixed_effects(Shat_level, demeaning_group)
    Dhat_fixed_effect = estimate_group_fixed_effects(Dhat_level, demeaning_group)

    _extra_cols = _compute_extra_instruments(context, Shat_level, Dhat_level, isolated, demeaning_group)  # [v5]
    _base = {
        "y": context["y"],
        "X": context["X"],
        "S": residualize_fixed_effects(S_level, S_fixed_effect),
        "Shat": residualize_fixed_effects(Shat_level, Shat_fixed_effect),
        "Dhat": residualize_fixed_effects(Dhat_level, Dhat_fixed_effect),
        "S_level": S_level,
        "Shat_level": Shat_level,
        "Dhat_level": Dhat_level,
        "S_fixed_effect": S_fixed_effect,
        "Shat_fixed_effect": Shat_fixed_effect,
        "Dhat_fixed_effect": Dhat_fixed_effect,
        "school_id": context["school_id"],
        "school_codes": context["school_codes"],
        "schools": context["schools"],
        "isolated": isolated,
    }
    _base.update(_extra_cols)  # [v5] configurable instrument columns
    return _base


def nonisolated_instruments(X_N, Shat_N, Dhat_N):
    """
    Build the instrument matrix for non-isolated students.

    z_i(beta) contains own covariates, the predicted peer norm, and the
    derivative of the predicted peer norm with respect to beta.
    """
    return np.column_stack([X_N, Shat_N, Dhat_N])


def build_equation_blocks(data):
    """
    Split estimation data into the two equations used by the moments.

    I: isolated students
    N: non-isolated students
    """
    isolated = data["isolated"]

    X_I = data["X"][isolated]
    y_I = data["y"][isolated]

    X_N = data["X"][~isolated]
    y_N = data["y"][~isolated]
    S_N = data["S"][~isolated]

    _extra_N = _extra_instrument_matrix(data)[~isolated]  # [v5] configurable instruments
    Z_N = np.column_stack([X_N, _extra_N]) if _extra_N.shape[1] else X_N

    if X_I.shape[0] == 0 or X_N.shape[0] == 0:
        raise ValueError(
            "The estimator needs both isolated and non-isolated students."
        )

    return {
        "X_I": X_I,
        "y_I": y_I,
        "X_N": X_N,
        "y_N": y_N,
        "S_N": S_N,
        "Z_N": Z_N,
        "N_I": X_I.shape[0],
        "N_N": X_N.shape[0],
    }


def isolated_residual(blocks, gamma):
    """Residual for isolated students: y_i - x_i' gamma."""
    return blocks["y_I"] - blocks["X_I"] @ gamma


def nonisolated_residual(blocks, gamma, lambda_value, delta_value):
    """Residual for non-isolated students."""
    return (
        blocks["y_N"]
        - delta_value * (blocks["X_N"] @ gamma)
        - lambda_value * blocks["S_N"]
    )


# ============================================================
# 3. Concentrate out gamma
# ============================================================

def identity_weights_for_covariates(k):
    """Create first-step identity weighting matrices."""
    return np.eye(k), np.eye(k + _n_extra_cols())  # [v5]


def identity_weights(data):
    """Create identity weighting matrices from an estimation data dictionary."""
    k = data["X"].shape[1]
    return identity_weights_for_covariates(k)


def default_weight_matrices(context):
    """
    Create the default first-step weighting matrices once.

    The isolated block has k moments. The non-isolated block has k covariate
    moments plus the active instruments, so it has k + _n_extra_cols() moments.
    """
    k = context["X"].shape[1]
    return identity_weights_for_covariates(k)


    # Math from the paper ("tedious but straightforward algebra..")
def solve_gamma_given_nonlinear_parameters(
    data,
    lambda_value,
    delta_value,
    W_I=None,
    W_N=None,
):
    """
    Given lambda, beta, and delta, solve analytically for gamma.

    The beta-dependent objects are already contained in data. The derivation is
    the standard linear-GMM concentration step applied to the isolated and
    non-isolated moment blocks.
    """
    blocks = build_equation_blocks(data)

    if W_I is None or W_N is None:
        W_I, W_N = identity_weights(data)

    X_I = blocks["X_I"]
    y_I = blocks["y_I"]
    X_N = blocks["X_N"]
    y_N = blocks["y_N"]
    S_N = blocks["S_N"]
    Z_N = blocks["Z_N"]
    N_I = blocks["N_I"]
    N_N = blocks["N_N"]

    isolated_gamma_jacobian = X_I.T @ X_I / N_I
    isolated_outcome_moment = X_I.T @ y_I / N_I

    nonisolated_gamma_jacobian = delta_value * Z_N.T @ X_N / N_N
    nonisolated_outcome_moment = Z_N.T @ (y_N - lambda_value * S_N) / N_N

    A = (
        isolated_gamma_jacobian.T @ W_I @ isolated_gamma_jacobian
        + nonisolated_gamma_jacobian.T @ W_N @ nonisolated_gamma_jacobian
    )
    b = (
        isolated_gamma_jacobian.T @ W_I @ isolated_outcome_moment
        + nonisolated_gamma_jacobian.T @ W_N @ nonisolated_outcome_moment
    )

    try:
        gamma_hat = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        gamma_hat = np.linalg.pinv(A) @ b

    return gamma_hat


def gmm_moments(data, gamma, lambda_value, delta_value):
    """Compute sample average moments for both equation groups."""
    blocks = build_equation_blocks(data)

    epsilon_I = isolated_residual(blocks, gamma)
    e_N = nonisolated_residual(
        blocks=blocks,
        gamma=gamma,
        lambda_value=lambda_value,
        delta_value=delta_value,
    )

    m_I = blocks["X_I"].T @ epsilon_I / blocks["N_I"]
    m_N = blocks["Z_N"].T @ e_N / blocks["N_N"]

    return m_I, m_N


def gmm_objective_given_data(
    data,
    gamma,
    lambda_value,
    delta_value,
    W_I=None,
    W_N=None,
):
    """Compute the weighted GMM objective."""
    if W_I is None or W_N is None:
        W_I, W_N = identity_weights(data)

    m_I, m_N = gmm_moments(data, gamma, lambda_value, delta_value)

    return float(m_I @ W_I @ m_I + m_N @ W_N @ m_N)


def estimate_weight_matrices(data, gamma, lambda_value, delta_value):
    """
    Estimate second-step GMM weighting matrices from first-step residuals.

    A tiny ridge might be added before inversion to keep stable when
    sample moments are nearly collinear.
    """
    blocks = build_equation_blocks(data)

    epsilon_I = isolated_residual(blocks, gamma)
    e_N = nonisolated_residual(
        blocks=blocks,
        gamma=gamma,
        lambda_value=lambda_value,
        delta_value=delta_value,
    )

    q_I = blocks["X_I"] * epsilon_I[:, None]
    q_N = blocks["Z_N"] * e_N[:, None]

    S_I = q_I.T @ q_I / blocks["N_I"]
    S_N_cov = q_N.T @ q_N / blocks["N_N"]

    W_I = np.linalg.pinv(S_I)
    W_N = np.linalg.pinv(S_N_cov)

    return W_I, W_N


# ============================================================
# 4. Cluster-robust standard errors
# ============================================================

def stack_weight_matrices(W_I, W_N):
    """Combine isolated and non-isolated weights into one block matrix."""
    q_I = W_I.shape[0]
    q_N = W_N.shape[0]

    W = np.zeros((q_I + q_N, q_I + q_N))
    W[:q_I, :q_I] = W_I
    W[q_I:, q_I:] = W_N

    return W


def theta_from_stage_results(stage_results):
    """Return [gamma, lambda, beta, delta] from one estimation stage."""
    return np.concatenate(
        [
            stage_results["gamma"],
            stage_results["nonlinear_parameters"],
        ]
    )


def stacked_moments_for_theta(theta, context):
    """Evaluate the stacked moment vector at a full parameter vector."""
    k = len(X_COLS)
    gamma = theta[:k]
    lambda_value, beta_value, delta_value = theta[k:]

    data = build_estimation_data(beta=beta_value, context=context)

    m_I, m_N = gmm_moments(
        data=data,
        gamma=gamma,
        lambda_value=lambda_value,
        delta_value=delta_value,
    )

    return np.concatenate([m_I, m_N])


def numerical_moment_jacobian(theta, context):
    """
    Numerically approximate d g(theta) / d theta'.

    The beta derivative is included by rebuilding beta-dependent peer norms at
    nearby beta values.
    """
    base_moments = stacked_moments_for_theta(theta, context)
    jacobian = np.zeros((len(base_moments), len(theta)))

    for j, value in enumerate(theta):
        step = max(abs(value) * 1e-5, 1e-5)

        theta_high = theta.copy()
        theta_low = theta.copy()
        theta_high[j] += step
        theta_low[j] -= step

        if PARAMETER_NAMES[j] == "beta" and theta_low[j] <= 0:
            theta_low[j] = value
            theta_high[j] = value + step
            moments_high = stacked_moments_for_theta(theta_high, context)
            jacobian[:, j] = (moments_high - base_moments) / step
            continue

        moments_high = stacked_moments_for_theta(theta_high, context)
        moments_low = stacked_moments_for_theta(theta_low, context)
        jacobian[:, j] = (moments_high - moments_low) / (2.0 * step)

    return jacobian


def school_cluster_moment_contributions(data, gamma, lambda_value, delta_value):
    """
    Aggregate moment contributions by school.

    Each row is one school's contribution to the stacked sample moments. The
    denominators match the moment definitions used in the GMM objective.
    """
    y = data["y"]
    X = data["X"]
    S = data["S"]
    school_codes = data["school_codes"]
    schools = data["schools"]
    isolated = data["isolated"]

    k = X.shape[1]
    extra_full = _extra_instrument_matrix(data)  # [v5]
    contributions = np.zeros((len(schools), 2 * k + extra_full.shape[1]))

    N_I = int(np.sum(isolated))
    N_N = int(np.sum(~isolated))

    epsilon_I = y - X @ gamma
    e_N = y - delta_value * (X @ gamma) - lambda_value * S

    for school_code in range(len(schools)):
        school_mask = school_codes == school_code

        isolated_mask = school_mask & isolated
        if np.any(isolated_mask):
            contributions[school_code, :k] = (
                X[isolated_mask].T @ epsilon_I[isolated_mask] / N_I
            )

        nonisolated_mask = school_mask & ~isolated
        if np.any(nonisolated_mask):
            Z_school = (np.column_stack([X[nonisolated_mask], extra_full[nonisolated_mask]])
                        if extra_full.shape[1] else X[nonisolated_mask])  # [v5]
            contributions[school_code, k:] = (
                Z_school.T @ e_N[nonisolated_mask] / N_N
            )

    return contributions


def covariance_for_reported_parameters(parameter_covariance):
    """Apply the delta method to lambda_1 and lambda_2."""
    rows = []

    for i in range(len(PARAMETER_NAMES)):
        row = np.zeros(len(PARAMETER_NAMES))
        row[i] = 1.0
        rows.append(row)

    lambda_1_row = np.zeros(len(PARAMETER_NAMES))
    lambda_1_row[3] = 1.0
    lambda_1_row[5] = 1.0
    rows.append(lambda_1_row)

    lambda_2_row = np.zeros(len(PARAMETER_NAMES))
    lambda_2_row[5] = -1.0
    rows.append(lambda_2_row)

    transform = np.vstack(rows)
    return transform @ parameter_covariance @ transform.T


def cluster_robust_standard_errors(context, results):
    """
    Compute school-cluster robust standard errors for the final GMM estimates.

    These SE use the final second-step weighting matrix and cluster the moment
    covariance at the school level.
    """
    final_stage = results["second_step"]
    theta_hat = theta_from_stage_results(final_stage)

    lambda_hat, beta_hat, delta_hat = final_stage["nonlinear_parameters"]
    gamma_hat = final_stage["gamma"]

    data_hat = build_estimation_data(beta=beta_hat, context=context)

    cluster_contributions = school_cluster_moment_contributions(
        data=data_hat,
        gamma=gamma_hat,
        lambda_value=lambda_hat,
        delta_value=delta_hat,
    )

    n_clusters = cluster_contributions.shape[0]
    small_sample_correction = n_clusters / (n_clusters - 1.0)  # [v2-C] G/(G-1)
    omega = small_sample_correction * (
        cluster_contributions.T @ cluster_contributions
    )

    jacobian = numerical_moment_jacobian(theta_hat, context)
    W = stack_weight_matrices(results["W_I"], results["W_N"])

    bread = jacobian.T @ W @ jacobian
    middle = jacobian.T @ W @ omega @ W @ jacobian
    bread_inv = np.linalg.pinv(bread)

    parameter_covariance = bread_inv @ middle @ bread_inv
    reported_covariance = covariance_for_reported_parameters(parameter_covariance)

    reported_names = PARAMETER_NAMES + DERIVED_PARAMETER_NAMES
    reported_se = np.sqrt(np.maximum(np.diag(reported_covariance), 0.0))

    return {
        "parameter_covariance": parameter_covariance,
        "reported_covariance": reported_covariance,
        "reported_se": pd.Series(reported_se, index=reported_names),
        "n_clusters": int(n_clusters),
    }


# ============================================================
# 5. Outer objective over lambda, beta, delta
# ============================================================

def concentrated_gmm_objective(params, context, W_I=None, W_N=None):
    """
    Objective minimized by scipy.optimize.minimize.

    The optimizer searches over lambda, beta, and delta. For each candidate,
    gamma is solved analytically.
    """
    lambda_value, beta_value, delta_value = params

    if beta_value <= 0 or delta_value <= 0 or lambda_value < 0:
        return 1e12

    try:
        # first we compute norm related quantities
        data = build_estimation_data(beta=beta_value, context=context)
        # then we get gamma given other params
        gamma_hat = solve_gamma_given_nonlinear_parameters(
            data=data,
            lambda_value=lambda_value,
            delta_value=delta_value,
            W_I=W_I,
            W_N=W_N,
        )
        #finally we compute the minimand
        return gmm_objective_given_data(
            data=data,
            gamma=gamma_hat,
            lambda_value=lambda_value,
            delta_value=delta_value,
            W_I=W_I,
            W_N=W_N,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return 1e12


def estimate_with_weights(
    context,
    W_I=None,
    W_N=None,
    starting_values=None,
    stage="first_step",
):
    """Run the concentrated GMM search for fixed weighting matrices."""
    if starting_values is None:
        starting_values = STARTING_VALUES

    if W_I is None or W_N is None:
        W_I, W_N = default_weight_matrices(context)

    optimization_results = []

    for start in starting_values:
        result = minimize(
            concentrated_gmm_objective,
            x0=start,
            args=(context, W_I, W_N),
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={
                "maxiter": 300,
                "ftol": 1e-10,
            },
        )
        optimization_results.append(result)

    best_result = min(optimization_results, key=lambda r: r.fun)

    lambda_hat, beta_hat, delta_hat = best_result.x

    data_hat = build_estimation_data(beta=beta_hat, context=context)

    gamma_hat = solve_gamma_given_nonlinear_parameters(
        data=data_hat,
        lambda_value=lambda_hat,
        delta_value=delta_hat,
        W_I=W_I,
        W_N=W_N,
    )

    estimates = {
        "gamma_age": gamma_hat[0],
        "gamma_female": gamma_hat[1],
        "gamma_f_col": gamma_hat[2],
        "lambda": lambda_hat,
        "beta": beta_hat,
        "delta": delta_hat,
        "lambda_1": lambda_hat + delta_hat - 1.0,
        "lambda_2": 1.0 - delta_hat,
        "objective": best_result.fun,
        "success": bool(best_result.success),
        "message": str(best_result.message),
        "iterations": int(best_result.nit),
    }

    return {
        "stage": stage,
        "data": data_hat,
        "gamma": gamma_hat,
        "nonlinear_parameters": best_result.x,
        "estimates": estimates,
        "optimizer": best_result,
    }


def estimate_model_two_step(df, G_list, compute_robust=True):  # [v2] compute_robust toggles the heavy AR/LIM
    """
    Estimate the model using two-step concentrated GMM.

    The first step uses identity matrices. The second step estimates the
    covariance of the moment contributions at the first-step estimate and uses
    its inverse as the weighting matrix.
    """
    context = prepare_estimation_context(df, G_list)

    # [v2-B] global beta pre-scan to seed the first step.
    prescan_start = beta_grid_prescan(context)

    first_step = estimate_with_weights(
        context=context,
        starting_values=[prescan_start, *STARTING_VALUES],
        stage="first_step_identity",
    )

    lambda_1, beta_1, delta_1 = first_step["nonlinear_parameters"]
    W_I, W_N = estimate_weight_matrices(
        data=first_step["data"],
        gamma=first_step["gamma"],
        lambda_value=lambda_1,
        delta_value=delta_1,
    )

    second_starting_values = [
        first_step["nonlinear_parameters"],
        *STARTING_VALUES,
    ]

    second_step = estimate_with_weights(
        context=context,
        W_I=W_I,
        W_N=W_N,
        starting_values=second_starting_values,
        stage="second_step_weighted",
    )

    results = {
        "first_step": first_step,
        "second_step": second_step,
        "W_I": W_I,
        "W_N": W_N,
        "yhat": context["yhat"],
        "n_observations": int(len(context["y"])),
        "n_clusters": int(len(context["schools"])),
    }

    results["cluster_robust_se"] = cluster_robust_standard_errors(
        context=context,
        results=results,
    )

    results["instrument_strength"] = instrument_strength(
        context, second_step["nonlinear_parameters"][1]
    )

    # [v2] AR + LIM are expensive (a beta grid each); skip them in Monte Carlo
    #      and the identification frontier, where only SE(beta) is needed.
    if compute_robust:
        # [v2-A] weak-IV-robust (Anderson-Rubin) confidence set for beta.
        results["anderson_rubin_beta"] = anderson_rubin_ci_beta(
            context, results["W_I"], results["W_N"], n_obs=results["n_observations"]
        )
        # [v2-F] linear-in-means (beta = 1) restriction for a nonlinearity test.
        results["lim_comparison"] = estimate_lim(context, results["W_I"], results["W_N"])

    return results


# ============================================================
# 6. Reporting
# ============================================================

def instrument_strength(context, beta):
    """Partial first-stage F for the beta instrument Dhat, given [X, Shat]."""
    data = build_estimation_data(beta=beta, context=context)
    N = ~data["isolated"]
    S = data["S"][N]
    X = data["X"][N]
    Sh = data["Shat"][N]
    Dh = data["Dhat"][N]

    def rss(Z, t):
        coef, *_ = np.linalg.lstsq(Z, t, rcond=None)
        resid = t - Z @ coef
        return float(resid @ resid)

    n = len(S)
    full = np.column_stack([X, Sh, Dh])
    base = np.column_stack([X, Sh])
    k = full.shape[1]
    F_beta = ((rss(base, S) - rss(full, S)) / 1) / (rss(full, S) / (n - k))

    # [v2-A] relevance/exogeneity tension: a near-perfect first stage means
    # Shat -> S and the instrument loses exogeneity.
    y_lvl, yh = context["y_level"], context["yhat"]
    yhat_r2 = float(1 - np.sum((y_lvl - yh) ** 2) / np.sum((y_lvl - y_lvl.mean()) ** 2))
    if yhat_r2 > 0.95:
        warnings.warn(
            f"First-stage R^2(yhat)={yhat_r2:.3f} is very high; Shat approaches "
            "the endogenous S and may lose exogeneity.", UserWarning)

    return {
        "F_beta_instruments": float(F_beta),
        "first_stage_R2": float(1 - rss(full, S) / float(S @ S)),
        "yhat_first_stage_R2": yhat_r2,
    }


# [v2-B] -----------------------------------------------------------------
def beta_grid_prescan(context, betas=None, lam=0.25, delta=0.85):
    """Coarse 1-D scan of the concentrated objective over beta (identity
    weights) to choose a good starting value (avoids the spurious beta ~ 2)."""
    if betas is None:
        betas = np.linspace(1.5, 40.0, 24)
    best_beta, best_obj = float(betas[0]), np.inf
    for b in betas:
        obj = concentrated_gmm_objective([lam, float(b), delta], context)
        if obj < best_obj:
            best_obj, best_beta = obj, float(b)
    return np.array([lam, best_beta, delta])


# [v2-A] -----------------------------------------------------------------
def _profile_objective_at_beta(beta0, context, W_I, W_N):
    """Min GMM objective over (lambda, delta) (gamma concentrated) at fixed beta0."""
    data = build_estimation_data(beta=beta0, context=context)

    def inner(ld):
        lam, dlt = ld
        if not (0 < lam < 1 and 0 < dlt < 1):
            return 1e12
        g = solve_gamma_given_nonlinear_parameters(data, lam, dlt, W_I, W_N)
        return gmm_objective_given_data(data, g, lam, dlt, W_I, W_N)

    best = np.inf
    for x0 in ([0.25, 0.85], [0.15, 0.80], [0.35, 0.90]):
        r = minimize(inner, x0, method="L-BFGS-B",
                     bounds=[(1e-3, 1 - 1e-3), (1e-3, 1 - 1e-3)])
        best = min(best, float(r.fun))
    return best


def anderson_rubin_ci_beta(context, W_I, W_N, betas=None, n_obs=None, alpha=0.05):
    """Approximate weak-IV-robust (Anderson-Rubin-style) confidence set for beta:
    the grid of beta whose over-identified J = n * Q is not rejected. Valid even
    when beta's instrument is weak (unlike the Wald t)."""
    from scipy.stats import chi2
    if betas is None:
        betas = np.linspace(1.5, 45.0, 30)
    n_moments = 2 * len(X_COLS) + _n_extra_cols()  # [v5]
    n_free = len(X_COLS) + 2
    crit = float(chi2.ppf(1 - alpha, n_moments - n_free))
    Js = [n_obs * _profile_objective_at_beta(float(b), context, W_I, W_N) for b in betas]
    inside = [float(b) for b, j in zip(betas, Js) if j <= crit]
    return {
        "betas": [float(b) for b in betas], "J": [float(j) for j in Js],
        "critical_value": crit,
        "ci_low": (min(inside) if inside else None),
        "ci_high": (max(inside) if inside else None),
        "note": "Approximate AR-style set (J = n_obs * concentrated GMM objective).",
    }


# [v2-F] -----------------------------------------------------------------
def estimate_lim(context, W_I, W_N):
    """Restricted linear-in-means fit (beta = 1); compare its objective with the
    free-beta fit as a test of nonlinearity."""
    data = build_estimation_data(beta=1.0, context=context)

    def inner(ld):
        lam, dlt = ld
        if not (0 < lam < 1 and 0 < dlt < 1):
            return 1e12
        g = solve_gamma_given_nonlinear_parameters(data, lam, dlt, W_I, W_N)
        return gmm_objective_given_data(data, g, lam, dlt, W_I, W_N)

    best = None
    for x0 in ([0.25, 0.85], [0.15, 0.80], [0.35, 0.90]):
        r = minimize(inner, x0, method="L-BFGS-B",
                     bounds=[(1e-3, 1 - 1e-3), (1e-3, 1 - 1e-3)])
        if best is None or r.fun < best.fun:
            best = r
    lam, dlt = best.x
    return {"beta": 1.0, "lambda": float(lam), "delta": float(dlt),
            "objective": float(best.fun),
            "note": "Restricted beta=1 (LIM); compare objective vs free-beta fit."}


# [v2-C] -----------------------------------------------------------------
def monte_carlo_v5(n_reps=50, n_schools=30, base_seed=20240):
    """Monte Carlo over independent DGP draws: bias, RMSE and 95% coverage.
    Heavy; reduce n_schools/n_reps for speed."""
    import DGP_v5 as _D
    names = PARAMETER_NAMES + DERIVED_PARAMETER_NAMES
    est = {n: [] for n in names}; cov = {n: 0 for n in names}; true = None
    for r in range(n_reps):
        base = int(np.random.SeedSequence(base_seed + r).generate_state(1)[0])
        _, Gl = _D.generate_all_networks(n_schools=n_schools, seed=base)
        df = _D.generate_students_and_covariates(n_schools=n_schools, seed=base + 1)
        df, true = _D.generate_gpa_from_model(df=df, G_list=Gl, seed=base + 2)
        res = estimate_model_two_step(df, Gl, compute_robust=False)  # [v2] fast
        s = res["second_step"]["estimates"]; se = res["cluster_robust_se"]["reported_se"]
        for n in names:
            est[n].append(s[n]); lo, hi = s[n] - 1.96 * se[n], s[n] + 1.96 * se[n]
            cov[n] += int(lo <= true[n] <= hi)
    rows = {}
    for n in names:
        a = np.array(est[n])
        rows[n] = {"true": true[n], "mean": float(a.mean()),
                   "bias": float(a.mean() - true[n]),
                   "rmse": float(np.sqrt(((a - true[n]) ** 2).mean())),
                   "coverage95": cov[n] / n_reps}
    return pd.DataFrame(rows).T


# [v2-F] -----------------------------------------------------------------
def robustness_sweep(n_schools=30, base_seed=777):
    """Re-estimate while varying p_isolated, sigma_epsilon and network density."""
    import DGP_v5 as _D
    rows = []
    for key, vals in {"p_isolated": [0.1, 0.2, 0.3],
                      "sigma_epsilon": [0.1, 0.2, 0.3],
                      "max_friends": [3, 5, 8]}.items():
        for v in vals:
            netkw = {"n_schools": n_schools, "seed": base_seed}
            if key in ("p_isolated", "max_friends"):
                netkw[key] = v
            _, Gl = _D.generate_all_networks(**netkw)
            df = _D.generate_students_and_covariates(n_schools=n_schools, seed=base_seed + 1)
            gkw = {"sigma_epsilon": v} if key == "sigma_epsilon" else {}
            df, _ = _D.generate_gpa_from_model(df=df, G_list=Gl, seed=base_seed + 2, **gkw)
            s = estimate_model_two_step(df, Gl, compute_robust=False)["second_step"]["estimates"]  # [v2] fast
            rows.append({"setting": key, "value": v, "lambda": s["lambda"],
                         "beta": s["beta"], "delta": s["delta"]})
    return pd.DataFrame(rows)


# [v2-F] -----------------------------------------------------------------
def planner_marginal_value(G, y, lam, delta, beta, isolated):
    """Marginal social value of subsidizing each student.

    The equilibrium is y = D p + lam * S(beta; y), so the total response to a
    private-component bump is dy/dp = (I - lam * J_S)^{-1} D, with the CES-norm
    Jacobian J_S[i,k] = g_ik * y_k^(beta-1) * A_i^(1/beta - 1) and D = diag(d_i)
    where d_i = delta for connected students and 1 for isolated ones.

        v_k = d(sum_i y_i)/dp_k = d_k * [ (I - lam * J_S)^{-T} 1 ]_k

    Under LIM (beta = 1) J_S = G (uniform); under the general model (beta > 1)
    influence scales with y^(beta-1), concentrating on high-outcome key players.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    G_dense = G.toarray() if sparse.issparse(G) else np.asarray(G, dtype=float)

    A = G_dense @ (y ** beta)
    active = ~isolated
    scale_i = np.zeros(n)
    scale_i[active] = A[active] ** (1.0 / beta - 1.0)
    J_S = (scale_i[:, None] * G_dense) * (y[None, :] ** (beta - 1.0))

    M = np.eye(n) - lam * J_S
    d = np.where(isolated, 1.0, delta)
    return d * np.linalg.solve(M.T, np.ones(n))


# [v2-C] -----------------------------------------------------------------
def identification_frontier(n_schools_grid=(20, 40, 80, 150), n_reps=10, base_seed=4321):
    """Precision of beta against design: for each number of schools, run n_reps
    DGP draws and record the mean SE(beta), RMSE(beta) and instrument F. Answers
    'how much data / how dense a network is needed to identify beta?'.
    Uses compute_robust=False, so it skips the (heavy) AR/LIM steps."""
    import DGP_v5 as _D
    rows = []
    for ns in n_schools_grid:
        ses, betas, fs = [], [], []
        true_beta = None
        for r in range(n_reps):
            base = int(np.random.SeedSequence(base_seed + 1000 * ns + r).generate_state(1)[0])
            _, Gl = _D.generate_all_networks(n_schools=ns, seed=base)
            df = _D.generate_students_and_covariates(n_schools=ns, seed=base + 1)
            df, tp = _D.generate_gpa_from_model(df=df, G_list=Gl, seed=base + 2)
            true_beta = tp["beta"]
            res = estimate_model_two_step(df, Gl, compute_robust=False)
            ses.append(res["cluster_robust_se"]["reported_se"]["beta"])
            betas.append(res["second_step"]["estimates"]["beta"])
            fs.append(res["instrument_strength"]["F_beta_instruments"])
        betas = np.array(betas)
        rows.append({"n_schools": ns,
                     "mean_SE_beta": float(np.mean(ses)),
                     "median_SE_beta": float(np.median(ses)),
                     "rmse_beta": float(np.sqrt(np.mean((betas - true_beta) ** 2))),
                     "mean_F": float(np.mean(fs))})
    return pd.DataFrame(rows)


def comparison_table(true_parameters, results):
    """Build a side-by-side table of true, first-step, and second-step values."""
    comparison = pd.concat(
        [
            pd.Series(true_parameters, name="true"),
            pd.Series(results["first_step"]["estimates"], name="first_step"),
            pd.Series(results["second_step"]["estimates"], name="second_step"),
        ],
        axis=1,
    )

    return comparison


def final_estimation_table(true_parameters, results):
    """Table with true values, final estimates, and cluster-robust SE."""
    estimates = pd.Series(results["second_step"]["estimates"], name="estimate")
    standard_errors = results["cluster_robust_se"]["reported_se"].rename(
        "cluster_se"
    )
    true_values = pd.Series(true_parameters, name="true")

    table = pd.concat([true_values, estimates, standard_errors], axis=1)
    table["t_stat"] = table["estimate"] / table["cluster_se"]

    return table.loc[PARAMETER_NAMES + DERIVED_PARAMETER_NAMES]


def json_ready(value):
    """Convert numpy objects to plain Python objects before writing JSON."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def save_estimation_outputs(results, comparison, output_dir=OUTPUT_DIR):
    """Save final estimation tables for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "estimation_comparison.csv")
    final_table = final_estimation_table(
        true_parameters=comparison["true"],
        results=results,
    )
    final_table.to_csv(output_dir / "final_estimates_with_cluster_se.csv")

    clean_results = {
        "first_step": results["first_step"]["estimates"],
        "second_step": results["second_step"]["estimates"],
        "cluster_robust_se": {
            "n_clusters": results["cluster_robust_se"]["n_clusters"],
            "reported_se": results["cluster_robust_se"]["reported_se"].to_dict(),
        },
        "n_observations": results["n_observations"],
        "n_clusters": results["n_clusters"],
        # [v2] persist the new diagnostics so figures/reports can read them
        "instrument_strength": results.get("instrument_strength"),
        "anderson_rubin_beta": results.get("anderson_rubin_beta"),
        "lim_comparison": results.get("lim_comparison"),
    }

    with (output_dir / "estimation_results.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(clean_results), f, indent=2)


def format_comparison_for_console(comparison):
    """Return a compact numeric table for classroom console output."""
    display = comparison.loc[REPORT_ROWS].apply(pd.to_numeric, errors="coerce")
    return display.round(4)


def format_final_table_for_console(final_table):
    """Return a compact final table with cluster-robust SE."""
    display = final_table.apply(pd.to_numeric, errors="coerce")
    return display.round(4)


def bound_warnings(estimates, tolerance=1e-6):
    """Report parameters that land on search bounds."""
    warnings = []

    for name, (lower, upper) in zip(["lambda", "beta", "delta"], BOUNDS):
        value = estimates[name]

        if abs(value - lower) <= tolerance:
            warnings.append(f"{name} is at its lower bound ({lower}).")

        if abs(value - upper) <= tolerance:
            warnings.append(f"{name} is at its upper bound ({upper}).")

    return warnings


def print_bound_warnings(results):
    """Print a short note when the optimizer chooses a boundary solution."""
    all_warnings = {
        stage_name: bound_warnings(stage_results["estimates"])
        for stage_name, stage_results in [
            ("first step", results["first_step"]),
            ("second step", results["second_step"]),
        ]
    }

    active = {
        stage_name: warnings
        for stage_name, warnings in all_warnings.items()
        if len(warnings) > 0
    }

    if len(active) == 0:
        return

    print()
    print("Bound check:")

    for stage_name, warnings in active.items():
        for warning in warnings:
            print(f"- {stage_name}: {warning}")


def main():
    df, _, G_list, true_parameters = load_environment(DATA_DIR)

    results = estimate_model_two_step(df, G_list)
    comparison = comparison_table(true_parameters, results)
    save_estimation_outputs(results, comparison)

    print("Concentrated two-step GMM estimates")
    print()
    print(format_comparison_for_console(comparison))
    print()
    print("Final estimates with school-cluster robust SE")
    print()
    final_table = final_estimation_table(true_parameters, results)
    print(format_final_table_for_console(final_table))
    print()
    print(
        "Clusters used for SE:",
        results["cluster_robust_se"]["n_clusters"],
    )
    print_bound_warnings(results)
    print()
    print(f"Saved estimation outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# ============================================================
# 6. [v3] Continuously Updated Estimator (CUE)
# ============================================================
# CUE difiere del two-step en UNA cosa: la matriz de pesos optima
# W(theta) = Omega(theta)^{-1} se RECALCULA en cada theta candidato dentro del
# objetivo, en vez de fijarse tras el primer paso. Elimina la dependencia del
# resultado respecto de los pesos del primer paso y suele bajar el sesgo de
# muestra finita cuando un instrumento es debil (aqui, beta).
# Referencia: Hansen, Heaton y Yaron (1996).

def cue_concentrated_objective(params, context, n_inner=3):  # [v3]
    """Objetivo CUE sobre (lambda, beta, delta). En cada theta: (i) construye
    los objetos que dependen de beta, (ii) itera gamma <-> W unas pocas veces
    para que gamma sea la solucion GLS bajo los MISMOS pesos optimos W(theta),
    y (iii) devuelve el objetivo GMM evaluado con ese W(theta). Recalcular W
    dentro del objetivo es lo que lo hace CUE y no two-step."""
    lambda_value, beta_value, delta_value = params
    if beta_value <= 0 or delta_value <= 0 or lambda_value < 0:
        return 1e12
    try:
        data = build_estimation_data(beta=beta_value, context=context)
        W_I, W_N = identity_weights(data)  # [v3] arranque identidad
        gamma_hat = solve_gamma_given_nonlinear_parameters(
            data=data, lambda_value=lambda_value, delta_value=delta_value, W_I=W_I, W_N=W_N)
        for _ in range(n_inner):  # [v3] consistencia gamma <-> W en este theta
            W_I, W_N = estimate_weight_matrices(data, gamma_hat, lambda_value, delta_value)
            gamma_hat = solve_gamma_given_nonlinear_parameters(
                data=data, lambda_value=lambda_value, delta_value=delta_value, W_I=W_I, W_N=W_N)
        W_I, W_N = estimate_weight_matrices(data, gamma_hat, lambda_value, delta_value)  # [v3] W en theta
        return gmm_objective_given_data(
            data=data, gamma=gamma_hat, lambda_value=lambda_value,
            delta_value=delta_value, W_I=W_I, W_N=W_N)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return 1e12


def _cue_solution_at(params, context, n_inner=3):  # [v3] recupera gamma y W en un theta
    lambda_value, beta_value, delta_value = params
    data = build_estimation_data(beta=beta_value, context=context)
    W_I, W_N = identity_weights(data)
    gamma_hat = solve_gamma_given_nonlinear_parameters(
        data=data, lambda_value=lambda_value, delta_value=delta_value, W_I=W_I, W_N=W_N)
    for _ in range(n_inner):
        W_I, W_N = estimate_weight_matrices(data, gamma_hat, lambda_value, delta_value)
        gamma_hat = solve_gamma_given_nonlinear_parameters(
            data=data, lambda_value=lambda_value, delta_value=delta_value, W_I=W_I, W_N=W_N)
    return data, gamma_hat, W_I, W_N


def estimate_cue(context, starting_values=None, n_inner=3):  # [v3]
    """Minimiza el objetivo CUE sobre (lambda, beta, delta) con multistart."""
    if starting_values is None:
        starting_values = STARTING_VALUES
    runs = []
    for start in starting_values:
        runs.append(minimize(
            cue_concentrated_objective, x0=start, args=(context, n_inner),
            method="L-BFGS-B", bounds=BOUNDS,
            options={"maxiter": 300, "ftol": 1e-10}))
    best = min(runs, key=lambda r: r.fun)
    lambda_hat, beta_hat, delta_hat = best.x
    data_hat, gamma_hat, W_I, W_N = _cue_solution_at(best.x, context, n_inner)
    estimates = {
        "gamma_age": gamma_hat[0], "gamma_female": gamma_hat[1], "gamma_f_col": gamma_hat[2],
        "lambda": lambda_hat, "beta": beta_hat, "delta": delta_hat,
        "lambda_1": lambda_hat + delta_hat - 1.0, "lambda_2": 1.0 - delta_hat,
        "objective": best.fun, "success": bool(best.success),
        "message": str(best.message), "iterations": int(best.nit)}
    return {"stage": "cue", "data": data_hat, "gamma": gamma_hat,
            "nonlinear_parameters": best.x, "estimates": estimates,
            "optimizer": best, "W_I": W_I, "W_N": W_N}


def estimate_model_cue(df, G_list, compute_robust=True, also_two_step=True, n_inner=3):  # [v3]
    """Estima por CUE. Mantiene la misma interfaz que estimate_model_two_step
    (CUE se expone como 'second_step') y, opcionalmente, corre tambien el
    two-step clasico para compararlos lado a lado."""
    context = prepare_estimation_context(df, G_list)
    prescan_start = beta_grid_prescan(context)
    first_step = estimate_with_weights(
        context=context, starting_values=[prescan_start, *STARTING_VALUES],
        stage="first_step_identity")
    cue = estimate_cue(
        context,
        starting_values=[first_step["nonlinear_parameters"], prescan_start, *STARTING_VALUES],
        n_inner=n_inner)
    results = {
        "first_step": first_step,
        "cue": cue,
        "second_step": cue,          # [v3] alias: CUE es el estimador final aguas abajo
        "estimator": "CUE",          # [v3]
        "W_I": cue["W_I"], "W_N": cue["W_N"],
        "yhat": context["yhat"],
        "n_observations": int(len(context["y"])),
        "n_clusters": int(len(context["schools"])),
    }
    if also_two_step:  # [v3] baseline two-step para comparar
        l1, b1, d1 = first_step["nonlinear_parameters"]
        W_I2, W_N2 = estimate_weight_matrices(first_step["data"], first_step["gamma"], l1, d1)
        results["two_step"] = estimate_with_weights(
            context=context, W_I=W_I2, W_N=W_N2,
            starting_values=[first_step["nonlinear_parameters"], *STARTING_VALUES],
            stage="two_step_weighted")
        results["two_step_se"] = cluster_robust_standard_errors(  # [v3] SE del two-step para comparar
            context, {"second_step": results["two_step"], "W_I": W_I2, "W_N": W_N2})
    results["cluster_robust_se"] = cluster_robust_standard_errors(context=context, results=results)
    results["instrument_strength"] = instrument_strength(context, cue["nonlinear_parameters"][1])
    if compute_robust:
        results["anderson_rubin_beta"] = anderson_rubin_ci_beta(
            context, results["W_I"], results["W_N"], n_obs=results["n_observations"])
        results["lim_comparison"] = estimate_lim(context, results["W_I"], results["W_N"])
    return results


def cue_vs_two_step_table(true_parameters, results):  # [v3]
    """Tabla verdadero / two-step / CUE para los parametros reportados."""
    names = PARAMETER_NAMES + DERIVED_PARAMETER_NAMES
    cols = {"true": pd.Series(true_parameters)}
    if "two_step" in results:
        cols["two_step"] = pd.Series(results["two_step"]["estimates"])
    cols["cue"] = pd.Series(results["cue"]["estimates"])
    table = pd.concat(cols, axis=1)
    keep = [n for n in names if n in table.index]
    return table.loc[keep]


def monte_carlo_cue_vs_two_step(n_reps=50, n_schools=30, base_seed=314159, n_inner=3):  # [v3]
    """Monte Carlo comparing the two-step and CUE estimators on independent DGP
    draws. Returns a (parameter x estimator) table with bias, RMSE and 95%
    coverage, so the two estimators can be put side by side. Heavy: each rep
    fits both estimators. Reduce n_reps / n_schools for speed."""
    import DGP_v5 as _D
    names = PARAMETER_NAMES + DERIVED_PARAMETER_NAMES
    est = {(tag, n): [] for tag in ("two_step", "cue") for n in names}
    cov = {(tag, n): 0 for tag in ("two_step", "cue") for n in names}
    true = None
    for r in range(n_reps):
        seed = int(np.random.SeedSequence(base_seed + r).generate_state(1)[0])
        _, Gl = _D.generate_all_networks(n_schools=n_schools, seed=seed)
        df = _D.generate_students_and_covariates(n_schools=n_schools, seed=seed + 1)
        df, true = _D.generate_gpa_from_model(df=df, G_list=Gl, seed=seed + 2)
        context = prepare_estimation_context(df, Gl)
        prescan = beta_grid_prescan(context)
        fs = estimate_with_weights(
            context, starting_values=[prescan, *STARTING_VALUES], stage="fs")
        l1, b1, d1 = fs["nonlinear_parameters"]
        # two-step
        W_I2, W_N2 = estimate_weight_matrices(fs["data"], fs["gamma"], l1, d1)
        ts = estimate_with_weights(
            context, W_I=W_I2, W_N=W_N2,
            starting_values=[fs["nonlinear_parameters"], *STARTING_VALUES], stage="ts")
        se_ts = cluster_robust_standard_errors(
            context, {"second_step": ts, "W_I": W_I2, "W_N": W_N2})["reported_se"]
        # CUE
        cue = estimate_cue(
            context, starting_values=[fs["nonlinear_parameters"], prescan], n_inner=n_inner)
        se_cue = cluster_robust_standard_errors(
            context, {"second_step": cue, "W_I": cue["W_I"], "W_N": cue["W_N"]})["reported_se"]
        for n in names:
            for tag, e, se in (("two_step", ts, se_ts), ("cue", cue, se_cue)):
                v = e["estimates"][n]; s = se[n]
                est[(tag, n)].append(v)
                cov[(tag, n)] += int(v - 1.96 * s <= true[n] <= v + 1.96 * s)
    rows = {}
    for n in names:
        for tag in ("two_step", "cue"):
            a = np.array(est[(tag, n)])
            rows[(n, tag)] = {
                "true": true[n], "mean": float(a.mean()),
                "bias": float(a.mean() - true[n]),
                "rmse": float(np.sqrt(((a - true[n]) ** 2).mean())),
                "coverage95": cov[(tag, n)] / n_reps}
    out = pd.DataFrame(rows).T
    out.index.set_names(["parameter", "estimator"], inplace=True)
    return out


def overid_j_test(n_obs, objective, n_moments=8, n_params=6):  # [v3]
    """Hansen over-identification J-test: J = n * Q ~ chi2(n_moments - n_params).
    Large J / small p would reject the over-identifying restrictions."""
    from scipy.stats import chi2
    dof = n_moments - n_params
    J = float(n_obs) * float(objective)
    return {"J": J, "dof": dof, "p_value": float(chi2.sf(J, dof))}


def format_full_report(true_parameters, results):  # [v3]
    """Rich console report for v3: point estimates (true / first / two-step / CUE),
    CUE final table with cluster SE, J-tests, instrument strength, Anderson-Rubin
    and the LIM restriction. Returns a string to print."""
    names = PARAMETER_NAMES + DERIVED_PARAMETER_NAMES
    out = []
    A = out.append
    A("=" * 70)
    A("  PROJECT REPLICATION v5   estimator: CUE   (two-step shown for comparison)")
    A("=" * 70)

    # [1] point estimates side by side
    cols = {"true": pd.Series(true_parameters),
            "first_step": pd.Series(results["first_step"]["estimates"])}
    if "two_step" in results:
        cols["two_step"] = pd.Series(results["two_step"]["estimates"])
    cols["CUE"] = pd.Series(results["cue"]["estimates"])
    pts = pd.concat(cols, axis=1).reindex(names)
    A("\n[1] Point estimates (true vs first-step vs two-step vs CUE)")
    A(pts.round(4).to_string())

    # [2] CUE final estimates with cluster SE
    A("\n[2] CUE final estimates with school-cluster robust SE")
    _ft = format_final_table_for_console(final_estimation_table(true_parameters, results))
    A(_ft if isinstance(_ft, str) else _ft.to_string())

    # [2b] 95% Wald confidence intervals and coverage of the truth  [v5]
    A("\n[2b] 95% Wald CIs (estimate +/- 1.96*SE) and contains-true check")
    _se = results["cluster_robust_se"]["reported_se"]; _est = results["cue"]["estimates"]
    for _nm in (PARAMETER_NAMES + DERIVED_PARAMETER_NAMES):
        _e = float(_est[_nm]); _s = float(_se[_nm]); _lo, _hi = _e - 1.96 * _s, _e + 1.96 * _s
        _tv = true_parameters.get(_nm)
        _ok = "yes" if (_tv is not None and _lo <= _tv <= _hi) else "NO"
        A(f"    {_nm:14s} [{_lo:8.4f}, {_hi:8.4f}]   true={_tv}   contains: {_ok}")

    # [3] beta head to head
    b_cue = results["cue"]["estimates"]["beta"]
    se_cue = results["cluster_robust_se"]["reported_se"]["beta"]
    line = f"\n[3] beta head-to-head:  CUE = {b_cue:.4f} (SE {se_cue:.4f})"
    if "two_step" in results and "two_step_se" in results:
        b_ts = results["two_step"]["estimates"]["beta"]
        se_ts = results["two_step_se"]["reported_se"]["beta"]
        line += f"   |   two-step = {b_ts:.4f} (SE {se_ts:.4f})"
    A(line)

    # [4] objective and over-identification J-test
    n = results["n_observations"]
    A("\n[4] GMM objective and over-identification J-test (dof = 8 - 6 = 2)")
    if "two_step" in results:
        q = results["two_step"]["estimates"]["objective"]; jt = overid_j_test(n, q, n_moments=2 * len(X_COLS) + _n_extra_cols())
        A(f"    two-step:  Q={q:.6f}  J=n*Q={jt['J']:.2f}  p={jt['p_value']:.3f}")
    q = results["cue"]["estimates"]["objective"]; jc = overid_j_test(n, q, n_moments=2 * len(X_COLS) + _n_extra_cols())
    A(f"    CUE:       Q={q:.6f}  J=n*Q={jc['J']:.2f}  p={jc['p_value']:.3f}")
    A("    (large J / small p -> the model's extra moments reject it)")

    # [5] instrument strength
    istr = results.get("instrument_strength")
    if istr:
        A("\n[5] Instrument strength for beta")
        A(f"    partial F = {istr.get('F_beta_instruments', float('nan')):.1f}"
          f"    first-stage R2 = {istr.get('first_stage_R2', float('nan')):.3f}")

    # [6] Anderson-Rubin
    ar = results.get("anderson_rubin_beta")
    if ar:
        A("\n[6] Anderson-Rubin 95% set for beta (weak-IV robust)")
        A(f"    [{ar['ci_low']}, {ar['ci_high']}]   (true beta = {true_parameters.get('beta')})")

    # [7] LIM restriction
    lim = results.get("lim_comparison")
    if lim:
        A("\n[7] LIM (beta = 1) restriction vs free")
        A(f"    objective LIM = {lim.get('objective', float('nan')):.6f}"
          f"   vs free CUE = {results['cue']['estimates']['objective']:.6f}")

    # [8] optimizer status
    A("\n[8] Optimizer status")
    c = results["cue"]["estimates"]
    A(f"    CUE:       success={c['success']}  iters={c['iterations']}")
    if "two_step" in results:
        t = results["two_step"]["estimates"]
        A(f"    two-step:  success={t['success']}  iters={t['iterations']}")
    A("=" * 70)
    return "\n".join(out)


def compare_instrument_sets(df, G_list, sets=None):  # [v5]
    """Refit the model (two-step) under several instrument transformations and
    report beta, its cluster SE, lambda, delta, the instrument F and the GMM
    objective for each. Shows which transformations identify beta worse."""
    global ACTIVE_INSTRUMENTS
    if sets is None:
        sets = {
            "baseline [X,S,D]":        ["Shat", "Dhat"],
            "+ D^2 (bad idea)":        ["Shat", "Dhat", "Dhat2"],
            "+ S^2":                   ["Shat", "Dhat", "Shat2"],
            "+ S(beta=1)":             ["Shat", "Dhat", "Shat_b1"],
            "+ friend covs GX":        ["Shat", "Dhat", "GX"],
            "+ friend-of-friend G2X":  ["Shat", "Dhat", "G2X"],
            "S only (beta unident.)":  ["Shat"],
        }
    saved = list(ACTIVE_INSTRUMENTS)
    rows = {}
    try:
        for name, instlist in sets.items():
            ACTIVE_INSTRUMENTS = list(instlist)
            try:
                n_instr = len(X_COLS) + _n_extra_cols()
                res = estimate_model_two_step(df, G_list, compute_robust=False)
                est = res["second_step"]["estimates"]
                se = res["cluster_robust_se"]["reported_se"]
                try:
                    F = float(res["instrument_strength"]["F_beta_instruments"])
                except Exception:
                    F = float("nan")
                rows[name] = {
                    "beta": round(float(est["beta"]), 3),
                    "se_beta": round(float(se["beta"]), 3),
                    "lambda": round(float(est["lambda"]), 3),
                    "delta": round(float(est["delta"]), 3),
                    "F_instr": round(F, 1),
                    "objective": float(est["objective"]),
                    "n_instr": n_instr,
                }
            except Exception as e:
                rows[name] = {"beta": float("nan"), "se_beta": float("nan"),
                              "lambda": float("nan"), "delta": float("nan"),
                              "F_instr": float("nan"), "objective": float("nan"),
                              "n_instr": -1, "note": str(e)[:30]}
    finally:
        ACTIVE_INSTRUMENTS = saved
    cols = ["beta", "se_beta", "lambda", "delta", "F_instr", "objective", "n_instr"]
    out = pd.DataFrame(rows).T
    return out.reindex(columns=[c for c in cols if c in out.columns] +
                       [c for c in out.columns if c not in cols])


def format_instrument_comparison(table, true_beta=10.0):  # [v5]
    """Console block for the instrument-set comparison."""
    lines = ["", "=" * 70,
             f" INSTRUMENT TRANSFORMATIONS  (true beta = {true_beta})", "=" * 70,
             table.to_string(), "-" * 70,
             " Read: beta far from true / huge se_beta / tiny F  =>  worse.",
             "=" * 70]
    return "\n".join(lines)
