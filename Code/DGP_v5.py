"""
Data-generating process (DGP) for the replication of
Boucher, Rendall, Ushchev & Zenou (2024) -- consolidated final version v5.

Builds the synthetic "classroom" data from the structural model so the estimator
can be validated against KNOWN true parameters.

    python DGP_v5.py            # writes data/generated_v5/

================================================================
WHAT CHANGED vs the primitive (DGP.py), with provenance tags
================================================================
[v2-D] Reproducibility: the network, the covariates and the outcome are drawn
       from THREE INDEPENDENT RNG streams (SeedSequence(set_seed).spawn(3))
       instead of one shared stream, removing spurious cross-block correlation.
[v1]   Realistic GPA scale: the private component p_i is RESAMPLED into [1, 4]
       (draw_private_in_range) and the intercept recalibrated (set_baseline
       2.5 -> 0.7) so the generated GPA is centered in [1, 4].
[v2-E] Institutional cap: the REALIZED equilibrium GPA is clipped to [1, 4]
       (set_clip_to_range=True) AFTER the fixed point converges -- never inside
       the iteration (that would redefine the estimated object). Because
       delta+lambda>1 creates a social multiplier, a few highly connected high
       types can land just above 4; n_gpa_clipped / share_gpa_clipped report how
       many were capped (here 1 of 30000).
[v2-D] Fixed-point diagnostics: per-school iteration counts + convergence flag,
       summarized as avg/max_fixed_point_iterations.
[v2-D] set_strict_range: optional rescaling of (delta, lambda) to sum <= 1 for a
       hard [1,4] guarantee (off by default; it would also kill the spillover
       lambda_1 = lambda + delta - 1).

set_seed drives all three streams; other modules import these settings.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # [v5-org] repo root (code in Code/)
DATA_DIR = PROJECT_ROOT / "Data" / "generated_v5"



# ============================================================
# 0. Checked DGP settings
# ============================================================

set_seed = 1234

# Network parameters
set_stu_perschool = 200
set_n_schools = 150
set_max_friends = 5
set_p_isolated = 0.2

# Private determinants
X_COLS = ["age", "female", "f_col"]
set_gamma = np.array([0.10, 0.24, 0.40])
set_baseline = 0.7  # bajado de 2.5: centra el GPA en [1,4] (solo nivel; no afecta lambda, delta, beta ni las pendientes)

set_sigma_epsilon = 0.20
set_sigma_school = 0.12

# Social interaction parameters
set_lambda_true = 0.25
set_delta_true = 0.85
set_beta_true = 10.0

# [v2-D] if True, rescale (delta, lambda) so delta+lambda <= 1 (GPA in [1,4] by construction)
set_strict_range = False

# [v2-E] if True, clip the realized equilibrium GPA to the [1,4] scale (institutional cap:
# a transcript GPA tops out at 4.0). Applied to the FINAL converged outcome only, never
# inside the fixed-point iteration (cf. the ces_norm note), so it does not redefine the
# object being estimated. With delta+lambda>1 the social multiplier can push a few highly
# connected high types just past 4; clipping that negligible share leaves the estimates
# unchanged (verified on the baseline draw: beta 9.3534 -> 9.3501, others identical).
set_clip_to_range = True


def settings_as_dict() -> dict[str, object]:
    """Return a JSON-friendly copy of the checked DGP settings."""
    return {
        "set_seed": set_seed,
        "set_stu_perschool": set_stu_perschool,
        "set_n_schools": set_n_schools,
        "set_max_friends": set_max_friends,
        "set_p_isolated": set_p_isolated,
        "set_gamma": set_gamma.tolist(),
        "set_baseline": set_baseline,
        "set_sigma_epsilon": set_sigma_epsilon,
        "set_sigma_school": set_sigma_school,
        "set_lambda_true": set_lambda_true,
        "set_delta_true": set_delta_true,
        "set_beta_true": set_beta_true,
        "set_strict_range": set_strict_range,  # [v2-D]
        "set_clip_to_range": set_clip_to_range,  # [v2-E]
    }


# ============================================================
# 1. Basic CES function
# ============================================================

# [v5-unify] CES kernel moved to core.py (single source of truth)
from core import ces_norm, peer_average


# ============================================================
# 2. Generate random directed friendship networks
# ============================================================

def generate_school_network(
    n_students=set_stu_perschool,
    max_friends=set_max_friends,
    p_isolated=set_p_isolated,
    rng=None,
):
    """
    Each student names up to max_friends friends.
    A share p_isolated names no friends.

    The network is directed. Rows are students who name friends, and columns
    are named friends.

    Returns:
        raw_G : unweighted adjacency matrix
        G     : row-normalized adjacency matrix
    """
    if rng is None:
        raise ValueError("No random number generator found. Set one first.")

    rows = []
    cols = []

    for i in range(n_students):
        if rng.random() < p_isolated:
            n_friends = 0
        else:
            n_friends = rng.integers(1, max_friends + 1)

        if n_friends == 0:
            continue

        possible_friends = np.delete(np.arange(n_students), i)
        friends = rng.choice(possible_friends, size=n_friends, replace=False)

        rows.extend([i] * n_friends)
        cols.extend(friends.tolist())

    raw_G = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_students, n_students),
    )

    row_sum = np.asarray(raw_G.sum(axis=1)).ravel()
    inv_row_sum = np.zeros_like(row_sum, dtype=float)
    inv_row_sum[row_sum > 0] = 1.0 / row_sum[row_sum > 0]

    G = (sparse.diags(inv_row_sum) @ raw_G).tocsr()

    return raw_G, G


def generate_all_networks(
    n_schools=set_n_schools,
    n_students_per_school=set_stu_perschool,
    max_friends=set_max_friends,
    p_isolated=set_p_isolated,
    seed=set_seed,
):
    """Generate one independent friendship network per school."""
    rng = np.random.default_rng(seed)

    raw_G_list = []
    G_list = []

    for _ in range(n_schools):
        raw_G, G = generate_school_network(
            n_students=n_students_per_school,
            max_friends=max_friends,
            p_isolated=p_isolated,
            rng=rng,
        )
        raw_G_list.append(raw_G)
        G_list.append(G)

    return raw_G_list, G_list


# ============================================================
# 3. Generate students and covariates
# ============================================================

def generate_students_and_covariates(
    n_schools=set_n_schools,
    n_students_per_school=set_stu_perschool,
    seed=set_seed,
):
    """Create the student-level covariates used by the DGP and estimation."""
    rng = np.random.default_rng(seed)

    rows = []
    student_id = 0

    for school_id in range(n_schools):
        for local_id in range(n_students_per_school):
            age = int(np.clip(np.rint(rng.normal(15.0, 1.2)), 13, 18))

            rows.append(
                {
                    "student_id": student_id,
                    "school_id": school_id,
                    "local_id": local_id,
                    "age": age,
                    "female": rng.binomial(1, 0.51),
                    "f_col": rng.binomial(1, 0.42),
                }
            )

            student_id += 1

    return pd.DataFrame(rows)


# ============================================================
# 4. Generate GPA from the model
# ============================================================

def generate_gpa_from_model(
    df,
    G_list,
    gamma=set_gamma,
    intercept=set_baseline,
    lambda_true=set_lambda_true,
    beta_true=set_beta_true,
    delta_true=set_delta_true,
    sigma_school=set_sigma_school,
    sigma_epsilon=set_sigma_epsilon,
    max_iter=1000,
    tolerance=1e-10,
    seed=set_seed,
):
    """
    Generate outcomes from the model.

    Private component:

        p_i = intercept + x_i' gamma + school effect + epsilon_i

    For isolated students:

        y_i = p_i

    For non-isolated students:

        y_i = delta * p_i + lambda * S_i(beta)

    where S_i(beta) is the CES norm of peers' outcomes.

    The CES social norm requires strictly positive entries. We keep the DGP in
    the safe region where p_i, lambda, and delta are positive.
    """
    if delta_true < 0:
        raise ValueError("delta_true must be nonnegative.")

    if lambda_true < 0:
        raise ValueError("lambda_true must be nonnegative.")

    if lambda_true >= 1.0:
        warnings.warn(
            "lambda_true >= 1. The peer feedback may be unstable.",
            UserWarning,
        )

    # [v2-D] strict GPA in [1,4] by construction: rescale so delta+lambda <= 1.
    if set_strict_range and (delta_true + lambda_true) > 1.0:
        scale = 1.0 / (delta_true + lambda_true)
        warnings.warn(
            "set_strict_range: rescaling (delta, lambda) to sum to 1 so the "
            f"outcome stays in [1, 4] by construction (scale={scale:.3f}). "
            "This sets the spillover lambda_1 to 0 (pure conformism).",
            UserWarning,
        )
        delta_true, lambda_true = delta_true * scale, lambda_true * scale

    def draw_private_in_range(mean, sd, rng, lo=1.0, hi=4.0):
        """
        Draw p_i = mean_i + epsilon_i, resampling epsilon_i only for
        observations where p_i falls outside [lo, hi].

        Restricting the private component keeps the GPA on the [1, 4] scale.
        We resample (rather than clip) so the structural equation still holds
        exactly for the generated data.
        """
        epsilon = rng.normal(loc=0.0, scale=sd, size=len(mean))
        private_component = mean + epsilon

        initially_bad = (private_component < lo) | (private_component > hi)
        n_initially_bad = int(np.sum(initially_bad))
        share_initially_bad = 100.0 * n_initially_bad / len(mean)

        while np.any((private_component < lo) | (private_component > hi)):
            bad = (private_component < lo) | (private_component > hi)
            epsilon[bad] = rng.normal(loc=0.0, scale=sd, size=int(np.sum(bad)))
            private_component[bad] = mean[bad] + epsilon[bad]

        if n_initially_bad > 0:
            warnings.warn(
                f"{share_initially_bad:.2f}% of the original epsilon draws "
                f"led to p_i outside [{lo}, {hi}], so those draws were resampled. "
                "If this percentage is large, consider changing the seed, gamma, "
                "or error variances.",
                UserWarning,
            )

        return private_component, epsilon, share_initially_bad

    rng = np.random.default_rng(seed)

    df = df.copy()
    X_all = df[X_COLS].to_numpy(float)

    y_all = np.empty(len(df))
    private_all = np.empty(len(df))
    epsilon_all = np.empty(len(df))

    school_effects = rng.normal(loc=0.0, scale=sigma_school, size=len(G_list))
    bad_draw_shares = []
    iteration_counts = []  # [v2-D] fixed-point iterations per school
    clipped_counts = []    # [v2-E] GPA values clipped to [1,4] per school

    start = 0

    for school_id, G in enumerate(G_list):
        n = G.shape[0]
        stop = start + n

        X = X_all[start:stop, :]
        private_mean = intercept + X @ gamma + school_effects[school_id]

        private_component, epsilon, bad_share = draw_private_in_range(
            mean=private_mean,
            sd=sigma_epsilon,
            rng=rng,
            lo=1.0,
            hi=4.0,
        )
        bad_draw_shares.append(bad_share)

        isolated = np.asarray(G.sum(axis=1)).ravel() == 0
        y = private_component.copy()

        converged = False
        for _it in range(max_iter):  # [v2-D]
            S = ces_norm(G, y, beta_true, isolated)

            y_new = delta_true * private_component + lambda_true * S
            y_new[isolated] = private_component[isolated]

            change = np.max(np.abs(y_new - y))
            y = y_new

            if change < tolerance:
                converged = True
                break

        if not converged:
            warnings.warn(
                f"School {school_id}: fixed point did not converge in "
                f"{max_iter} iterations (last change {change:.2e}).",
                UserWarning,
            )

        iteration_counts.append(_it + 1)  # [v2-D]

        if np.any(y <= 0):
            raise ValueError(
                "The fixed point produced non-positive outcomes. "
                "These cannot enter the CES norm. Check lambda_true, "
                "delta_true, and the private component."
            )

        # [v2-E] institutional cap on the *realized* GPA (a transcript tops out at 4.0).
        # Clipping is applied to the converged outcome, not the fixed-point operator
        # (cf. ces_norm note), so it does not redefine the object being estimated. With
        # delta_true + lambda_true > 1 the social multiplier can push a few highly
        # connected high types just past 4; this is a negligible share here.
        n_out_of_range = int(np.sum((y < 1.0) | (y > 4.0)))
        clipped_counts.append(n_out_of_range)
        if n_out_of_range > 0:
            if set_clip_to_range:
                y = np.clip(y, 1.0, 4.0)  # [v2-E] cap realized GPA to [1, 4]
            else:
                warnings.warn(
                    f"School {school_id}: {n_out_of_range} GPA value(s) fell outside "
                    "[1, 4] after the fixed point and were left unclipped because "
                    "set_clip_to_range is False. The social term can push connected "
                    "students beyond the private range when delta_true + lambda_true > 1.",
                    UserWarning,
                )

        y_all[start:stop] = y
        private_all[start:stop] = private_component
        epsilon_all[start:stop] = epsilon

        start = stop

    df["private_component"] = private_all
    df["epsilon"] = epsilon_all
    df["gpa"] = y_all

    true_parameters = {
        "gamma_age": gamma[0],
        "gamma_female": gamma[1],
        "gamma_f_col": gamma[2],
        "lambda": lambda_true,
        "beta": beta_true,
        "delta": delta_true,
        "lambda_1": lambda_true + delta_true - 1.0,
        "lambda_2": 1.0 - delta_true,
        "avg_initial_bad_epsilon_share": float(np.mean(bad_draw_shares)),
        "avg_fixed_point_iterations": float(np.mean(iteration_counts)),  # [v2-D]
        "max_fixed_point_iterations": int(np.max(iteration_counts)),  # [v2-D]
        "n_gpa_clipped": int(np.sum(clipped_counts)),  # [v2-E] obs clipped to [1,4]
        "share_gpa_clipped": float(np.sum(clipped_counts)) / len(df),  # [v2-E]
    }

    return df, true_parameters


# ============================================================
# 5. Reproducible project entry points
# ============================================================

def build_environment():
    """Build the full synthetic environment using independent RNG streams."""
    # [v2-D] independent RNG streams per block (network / covariates / outcome)
    seed_net, seed_cov, seed_out = (
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(set_seed).spawn(3)
    )

    raw_G_list, G_list = generate_all_networks(
        n_schools=set_n_schools,
        n_students_per_school=set_stu_perschool,
        max_friends=set_max_friends,
        p_isolated=set_p_isolated,
        seed=seed_net,
    )

    df = generate_students_and_covariates(
        n_schools=set_n_schools,
        n_students_per_school=set_stu_perschool,
        seed=seed_cov,
    )

    df, true_parameters = generate_gpa_from_model(
        df=df,
        G_list=G_list,
        gamma=set_gamma,
        intercept=set_baseline,
        lambda_true=set_lambda_true,
        beta_true=set_beta_true,
        delta_true=set_delta_true,
        sigma_school=set_sigma_school,
        sigma_epsilon=set_sigma_epsilon,
        seed=seed_out,
    )

    return df, raw_G_list, G_list, true_parameters


def save_environment(df, raw_G_list, G_list, true_parameters, output_dir=DATA_DIR):
    """Save generated data and networks so estimation can be rerun separately."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_network_file in output_dir.glob("*G_school_*.npz"):
        old_network_file.unlink(missing_ok=True)

    df.to_csv(output_dir / "students.csv", index=False)

    for school_id, raw_G in enumerate(raw_G_list):
        sparse.save_npz(output_dir / f"raw_G_school_{school_id:02d}.npz", raw_G)

    for school_id, G in enumerate(G_list):
        sparse.save_npz(output_dir / f"G_school_{school_id:02d}.npz", G)

    with (output_dir / "true_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(true_parameters, f, indent=2)

    metadata = {
        "project": "Project Replication Zenou",
        "settings": settings_as_dict(),
        "files": {
            "students": "students.csv",
            "true_parameters": "true_parameters.json",
            "raw_network_pattern": "raw_G_school_XX.npz",
            "network_pattern": "G_school_XX.npz",
        },
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_environment(input_dir=DATA_DIR):
    """Load the data and networks created by save_environment."""
    input_dir = Path(input_dir)
    student_file = input_dir / "students.csv"
    parameter_file = input_dir / "true_parameters.json"

    if not student_file.exists() or not parameter_file.exists():
        raise FileNotFoundError(
            "Generated DGP files were not found. Run `python DGP_v5.py` first."
        )

    df = pd.read_csv(student_file)

    def school_number(path):
        return int(path.stem.rsplit("_", maxsplit=1)[-1])

    raw_files = sorted(input_dir.glob("raw_G_school_*.npz"), key=school_number)
    G_files = sorted(input_dir.glob("G_school_*.npz"), key=school_number)

    if len(raw_files) == 0 or len(G_files) == 0:
        raise FileNotFoundError(
            "Network files were not found. Run `python DGP_v5.py` first."
        )

    raw_G_list = [sparse.load_npz(path).tocsr() for path in raw_files]
    G_list = [sparse.load_npz(path).tocsr() for path in G_files]

    with parameter_file.open("r", encoding="utf-8") as f:
        true_parameters = json.load(f)

    return df, raw_G_list, G_list, true_parameters


def summarize_environment(df, G_list, true_parameters):
    """Create a short text summary for classroom output."""
    isolated = np.concatenate(
        [np.asarray(G.sum(axis=1)).ravel() == 0 for G in G_list]
    )

    lines = [
        "Synthetic environment created.",
        "",
        f"Number of schools: {df['school_id'].nunique()}",
        f"Number of students: {len(df)}",
        f"Mean GPA: {df['gpa'].mean():.3f}",
        f"Min GPA: {df['gpa'].min():.3f}",
        f"Max GPA: {df['gpa'].max():.3f}",
        f"Share isolated: {np.mean(isolated):.3f}",
        "",
        "True parameters:",
        pd.Series(true_parameters).to_string(),
    ]

    return "\n".join(lines)


def main():
    df, raw_G_list, G_list, true_parameters = build_environment()
    save_environment(df, raw_G_list, G_list, true_parameters)

    print(summarize_environment(df, G_list, true_parameters))
    print()
    print(f"Saved generated data to: {DATA_DIR}")


if __name__ == "__main__":
    main()


# ============================================================
# [v5-ext] EXTENDED MODEL: contextual peer effects (phi'(Gx)) + heterogeneous
# schools, identified with peers-of-peers (G^2x) instruments. Absorbed here so
# the whole project lives under the v5 namespace (functions prefixed ext_).
# ============================================================
EXT_DATA_DIR = PROJECT_ROOT / "Data" / "generated_v5" / "extended"

# ----------------------------- settings ------------------------------
EXT_SEED = 2026
EXT_N_SCHOOLS = 150
EXT_MIN_STUDENTS = 80          # [v5-ext] heterogeneous school sizes
EXT_MAX_STUDENTS = 320         # [v5-ext]
EXT_MAX_FRIENDS = 6
EXT_P_ISOLATED = 0.18
EXT_GAMMA = np.array([0.10, 0.24, 0.40])   # own characteristics
EXT_PHI   = np.array([0.00, 0.15, 0.30])   # [v5-ext] contextual: peers' (age, female, parent-college)
EXT_LAMBDA_TRUE = 0.30                      # peer-norm intensity
EXT_BETA_TRUE = 5.0                         # CES curvature
EXT_BASELINE = -0.20
EXT_SIGMA_EPSILON = 0.20
EXT_SIGMA_SCHOOL = 0.12


# ----------------------------- CES core ------------------------------
# [v5-unify] CES kernel moved to core.py (single source of truth)


# ----------------------------- networks ------------------------------
def ext_generate_school_sizes(n_schools=EXT_N_SCHOOLS, seed=EXT_SEED):
    """[v5-ext] Draw a heterogeneous size for each school (realistic)."""
    rng = np.random.default_rng(seed)
    return rng.integers(EXT_MIN_STUDENTS, EXT_MAX_STUDENTS + 1, size=n_schools)


def ext_generate_school_network(n_students, max_friends, p_isolated, rng):
    rows, cols = [], []
    for i in range(n_students):
        n_friends = 0 if rng.random() < p_isolated else rng.integers(1, max_friends + 1)
        if n_friends == 0:
            continue
        friends = rng.choice(np.delete(np.arange(n_students), i), size=min(n_friends, n_students - 1), replace=False)
        rows.extend([i] * len(friends)); cols.extend(friends.tolist())
    raw_G = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_students, n_students))
    rs = np.asarray(raw_G.sum(axis=1)).ravel()
    inv = np.zeros_like(rs, float); inv[rs > 0] = 1.0 / rs[rs > 0]
    G = (sparse.diags(inv) @ raw_G).tocsr()
    return raw_G, G


def ext_generate_all_networks(sizes, max_friends=EXT_MAX_FRIENDS, p_isolated=EXT_P_ISOLATED, seed=EXT_SEED):
    """[v5-ext] One network per school, with per-school (heterogeneous) sizes."""
    rng = np.random.default_rng(seed)
    raw_G_list, G_list = [], []
    for n in sizes:
        raw_G, G = ext_generate_school_network(int(n), max_friends, p_isolated, rng)
        raw_G_list.append(raw_G); G_list.append(G)
    return raw_G_list, G_list


def ext_generate_students_and_covariates(sizes, seed=EXT_SEED):
    """[v5-ext] Student covariates; the number per school follows `sizes`."""
    rng = np.random.default_rng(seed)
    rows, sid = [], 0
    for school_id, n in enumerate(sizes):
        for local_id in range(int(n)):
            rows.append({"student_id": sid, "school_id": school_id, "local_id": local_id,
                         "age": int(np.clip(np.rint(rng.normal(15.0, 1.2)), 13, 18)),
                         "female": int(rng.random() < 0.5),
                         "f_col": int(rng.random() < 0.4)})
            sid += 1
    return pd.DataFrame(rows)


# ----------------------------- outcome -------------------------------
def _ext_draw_private_in_range(mean, sd, rng, lo=1.0, hi=4.0):
    eps = rng.normal(0.0, sd, size=len(mean))
    p = mean + eps
    bad0 = int(np.sum((p < lo) | (p > hi)))
    while np.any((p < lo) | (p > hi)):
        b = (p < lo) | (p > hi)
        p[b] = mean[b] + rng.normal(0.0, sd, size=int(np.sum(b)))
    return p, 100.0 * bad0 / len(mean)


def ext_generate_gpa_from_model(df, G_list, gamma=EXT_GAMMA, phi=EXT_PHI, lambda_true=EXT_LAMBDA_TRUE,
                            beta_true=EXT_BETA_TRUE, intercept=EXT_BASELINE,
                            sigma_school=EXT_SIGMA_SCHOOL, sigma_epsilon=EXT_SIGMA_EPSILON,
                            seed=EXT_SEED, max_iter=2000, tol=1e-10):
    """Outcome with the [v5-ext] contextual effect phi'(Gx) and a fixed point.

        connected: y_i = p_i + phi'(Gx)_i + lambda S_i(beta, y)
        isolated:  y_i = p_i,   with p_i = c + x_i'gamma + u_school + eps_i
    Realized GPA is clipped to [1,4]."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    X_all = df[X_COLS].to_numpy(float)
    y_all = np.empty(len(df)); p_all = np.empty(len(df))
    school_eff = rng.normal(0.0, sigma_school, size=len(G_list))
    n_clip_total = 0; start = 0
    for s, G in enumerate(G_list):
        n = G.shape[0]; sl = slice(start, start + n)
        X = X_all[sl]
        isolated = np.asarray(G.sum(axis=1)).ravel() == 0
        pmean = intercept + X @ gamma + school_eff[s]
        p, _ = _ext_draw_private_in_range(pmean, sigma_epsilon, rng, 1.0, 3.6)
        contextual = peer_average(G, X) @ phi          # [v5-ext] phi'(Gx)_i
        y = p.copy()
        for _ in range(max_iter):
            S = ces_norm(G, y, beta_true, isolated)
            y_new = p + contextual + lambda_true * S
            y_new[isolated] = p[isolated]
            if np.max(np.abs(y_new - y)) < tol:
                y = y_new; break
            y = y_new
        n_clip_total += int(np.sum((y < 1.0) | (y > 4.0)))
        y = np.clip(y, 1.0, 4.0)                        # realistic cap
        y_all[sl] = y; p_all[sl] = p; start += n
    df["gpa"] = y_all; df["private_component"] = p_all
    true_parameters = {
        "gamma_age": gamma[0], "gamma_female": gamma[1], "gamma_f_col": gamma[2],
        "phi_age": phi[0], "phi_female": phi[1], "phi_f_col": phi[2],
        "lambda": lambda_true, "beta": beta_true,
        "n_gpa_clipped": int(n_clip_total), "share_gpa_clipped": n_clip_total / len(df),
    }
    return df, true_parameters


# ----------------------------- entry points --------------------------
def ext_build_environment(seed=EXT_SEED):
    sizes = ext_generate_school_sizes(EXT_N_SCHOOLS, seed)
    raw_G_list, G_list = ext_generate_all_networks(sizes, seed=seed + 1)
    df = ext_generate_students_and_covariates(sizes, seed=seed + 2)
    df, true_parameters = ext_generate_gpa_from_model(df, G_list, seed=seed + 3)
    return df, raw_G_list, G_list, true_parameters


def ext_summarize_environment(df, G_list, true_parameters):
    sizes = df.groupby("school_id").size()
    iso = np.concatenate([np.asarray(G.sum(axis=1)).ravel() == 0 for G in G_list])
    lines = ["Synthetic environment created (extended model (contextual peer effects)).", "",
             f"Number of schools: {df['school_id'].nunique()}",
             f"Number of students: {len(df)}",
             f"School size: min {sizes.min()}, mean {sizes.mean():.1f}, max {sizes.max()}  [v5-ext: heterogeneous]",
             f"Mean GPA: {df['gpa'].mean():.3f}   range [{df['gpa'].min():.3f}, {df['gpa'].max():.3f}]",
             f"Share isolated: {np.mean(iso):.3f}", "",
             "True parameters:", pd.Series(true_parameters).to_string()]
    return "\n".join(lines)


def ext_save_environment(df, raw_G_list, G_list, true_parameters, output_dir=EXT_DATA_DIR):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "students.csv", index=False)
    for k, G in enumerate(G_list):
        sparse.save_npz(output_dir / f"G_school_{k:03d}.npz", G)
    for k, rg in enumerate(raw_G_list):                      # [v5-ext] persist raw networks too
        sparse.save_npz(output_dir / f"raw_G_school_{k:03d}.npz", rg)
    with (output_dir / "true_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(true_parameters, f, indent=2)


def ext_load_environment(input_dir=EXT_DATA_DIR):
    input_dir = Path(input_dir)
    df = pd.read_csv(input_dir / "students.csv")
    def num(p): return int(p.stem.rsplit("_", 1)[-1])
    G_list = [sparse.load_npz(p).tocsr() for p in sorted(input_dir.glob("G_school_*.npz"), key=num)]
    _rawf = sorted(input_dir.glob("raw_G_school_*.npz"), key=num)
    raw_G_list = [sparse.load_npz(p).tocsr() for p in _rawf] if _rawf else list(G_list)
    true_parameters = json.load(open(input_dir / "true_parameters.json"))
    return df, raw_G_list, G_list, true_parameters


