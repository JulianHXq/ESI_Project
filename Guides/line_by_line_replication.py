"""
Line-by-line classroom version of the Zenou replication routine.

The original project keeps a clean function-based structure; 
this version repeats more code on purpose so we can see the objects being created.

The only substantial function kept here is the objective passed to
scipy.optimize.minimize, because scipy needs a callable objective.
"""

# %%
# 0. Imports and paths

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
import matplotlib.pyplot as plt

if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    CWD = Path.cwd().resolve()
    if (CWD / "line_by_line_replication.py").exists():
        SCRIPT_DIR = CWD
    elif (CWD / "Code" / "LineByLine").exists():
        SCRIPT_DIR = CWD / "Code" / "LineByLine"
    elif (CWD / "LineByLine").exists():
        SCRIPT_DIR = CWD / "LineByLine"
    else:
        SCRIPT_DIR = CWD

CODE_DIR = SCRIPT_DIR.parent
if not (CODE_DIR / "DGP.py").exists() and (CODE_DIR.parent / "DGP.py").exists():
    CODE_DIR = CODE_DIR.parent
if not (CODE_DIR / "DGP.py").exists() and (Path.cwd() / "Code" / "DGP.py").exists():
    CODE_DIR = Path.cwd().resolve() / "Code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from DGP import (
    X_COLS,
    set_baseline,
    set_beta_true,
    set_delta_true,
    set_gamma,
    set_lambda_true,
    set_max_friends,
    set_n_schools,
    set_p_isolated,
    set_seed,
    set_sigma_epsilon,
    set_sigma_school,
    set_stu_perschool,
)


OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True)


# %%
# 1. Setup values

k = len(X_COLS)
gamma_names = [f"gamma_{name}" for name in X_COLS]
parameter_names = [*gamma_names, "lambda", "beta", "delta"]
derived_parameter_names = ["lambda_1", "lambda_2"]
reported_names = [*parameter_names, *derived_parameter_names]

starting_values = [
    np.array([0.10, 2.0, 0.10]),
    np.array([0.25, 8.0, 0.25]),
    np.array([0.50, 15.0, 0.50]),
]

bounds = [
    (0.001, 1.0),      # lambda
    (0.001, 400.0),    # beta
    (0.001, 1.0),      # delta
]

print("X columns:", X_COLS)
print("Number of schools:", set_n_schools)
print("Students per school:", set_stu_perschool)


# %%
# 2. Generate one directed friendship network per school

rng_network = np.random.default_rng(set_seed)

raw_G_list = []
G_list = []

for school_id in range(set_n_schools):
    rows = []
    cols = []

    for i in range(set_stu_perschool):
        if rng_network.random() < set_p_isolated:
            n_friends = 0
        else:
            n_friends = rng_network.integers(1, set_max_friends + 1)

        if n_friends == 0:
            continue

        possible_friends = np.delete(np.arange(set_stu_perschool), i)
        friends = rng_network.choice(
            possible_friends,
            size=n_friends,
            replace=False,
        )

        rows.extend([i] * n_friends)
        cols.extend(friends.tolist())

    raw_G = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(set_stu_perschool, set_stu_perschool),
    )

    row_sum = np.asarray(raw_G.sum(axis=1)).ravel()
    inv_row_sum = np.zeros_like(row_sum, dtype=float)
    inv_row_sum[row_sum > 0] = 1.0 / row_sum[row_sum > 0]

    G = (sparse.diags(inv_row_sum) @ raw_G).tocsr()

    raw_G_list.append(raw_G)
    G_list.append(G)

network_rows = []
for school_id, raw_G in enumerate(raw_G_list):
    out_degree = np.asarray(raw_G.sum(axis=1)).ravel()
    network_rows.append(
        {
            "school_id": school_id,
            "mean_friends": out_degree.mean(),
            "share_isolated": np.mean(out_degree == 0),
        }
    )

network_summary = pd.DataFrame(network_rows)
print(network_summary.head())

# %%
# 2bis. Plot Network

school_to_plot = 0
raw_G_plot = raw_G_list[school_to_plot].tocsr()
n_plot = raw_G_plot.shape[0]

angles = np.linspace(0, 2 * np.pi, n_plot, endpoint=False)
x_pos = np.cos(angles)
y_pos = np.sin(angles)

rows, cols = raw_G_plot.nonzero()

plt.figure(figsize=(7, 7))

# Draw directed friendship links
for i, j in zip(rows, cols):
    plt.arrow(
        x_pos[i],
        y_pos[i],
        0.99 * (x_pos[j] - x_pos[i]),
        0.99 * (y_pos[j] - y_pos[i]),
        length_includes_head=True,
        head_width=0.02,
        alpha=0.25,
        linewidth=0.8,
    )

# Draw students
out_degree = np.asarray(raw_G_plot.sum(axis=1)).ravel()
isolated_school = out_degree == 0

plt.scatter(
    x_pos[~isolated_school],
    y_pos[~isolated_school],
    s=60,
    label="Non-isolated",
)

plt.scatter(
    x_pos[isolated_school],
    y_pos[isolated_school],
    s=60,
    marker="x",
    label="Isolated",
)

for i in range(n_plot):
    plt.text(
        1.08 * x_pos[i],
        1.08 * y_pos[i],
        str(i),
        ha="center",
        va="center",
        fontsize=8,
    )

plt.title(f"Directed friendship network, school {school_to_plot}")
plt.axis("equal")
plt.axis("off")
plt.legend()
plt.show()


plt.figure(figsize=(8, 4))
plt.bar(network_summary["school_id"], network_summary["mean_friends"])
plt.xlabel("School")
plt.ylabel("Mean number of friends")
plt.title("Average out-degree by school")
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(network_summary["school_id"], network_summary["share_isolated"])
plt.xlabel("School")
plt.ylabel("Share isolated")
plt.title("Share of isolated students by school")
plt.ylim(0, 1)
plt.show()
 
# %%
# 3. Generate student covariates

rng_students = np.random.default_rng(set_seed)

student_rows = []
student_id = 0

for school_id in range(set_n_schools):
    for local_id in range(set_stu_perschool):
        age = int(np.clip(np.rint(rng_students.normal(15.0, 1.2)), 13, 18))

        student_rows.append(
            {
                "student_id": student_id,
                "school_id": school_id,
                "local_id": local_id,
                "age": age,
                "female": rng_students.binomial(1, 0.51),
                "f_col": rng_students.binomial(1, 0.42),
                "m_col": rng_students.binomial(1, 0.42),
            }
        )

        student_id += 1

df = pd.DataFrame(student_rows)
print(df.head())
print(df[X_COLS].describe())


# %%
# 4. Generate GPA from the model

if len(set_gamma) != len(X_COLS):
    raise ValueError("set_gamma must have one entry for each variable in X_COLS.")

rng_gpa = np.random.default_rng(set_seed)
school_effects = rng_gpa.normal(
    loc=0.0,
    scale=set_sigma_school,
    size=set_n_schools,
)

X_all = df[X_COLS].to_numpy(float)
y_all = np.empty(len(df))
private_component_all = np.empty(len(df))
epsilon_all = np.empty(len(df))
initial_bad_epsilon_shares = []

start = 0

for school_id, G in enumerate(G_list):
    n = G.shape[0]
    stop = start + n

    X_school = X_all[start:stop, :]
    private_mean = set_baseline + X_school @ set_gamma + school_effects[school_id]

    epsilon = rng_gpa.normal(
        loc=0.0,
        scale=set_sigma_epsilon,
        size=n,
    )
    private_component = private_mean + epsilon

    initially_bad = private_component <= 0
    initial_bad_epsilon_shares.append(np.mean(initially_bad))

    while np.any(private_component <= 0):
        bad = private_component <= 0
        epsilon[bad] = rng_gpa.normal(
            loc=0.0,
            scale=set_sigma_epsilon,
            size=np.sum(bad),
        )
        private_component[bad] = private_mean[bad] + epsilon[bad]

    isolated_school = np.asarray(G.sum(axis=1)).ravel() == 0
    y_school = private_component.copy()

    for iteration in range(1000):
        S_school = np.zeros_like(y_school)
        peer_sum = G @ (y_school ** set_beta_true)
        S_school[~isolated_school] = (
            peer_sum[~isolated_school] ** (1.0 / set_beta_true)
        )

        y_new = set_delta_true * private_component + set_lambda_true * S_school
        y_new[isolated_school] = private_component[isolated_school]

        change = np.max(np.abs(y_new - y_school))
        y_school = y_new

        if change < 1e-10:
            break

    if np.any(y_school <= 0):
        raise ValueError("The fixed point produced non-positive outcomes.")

    y_all[start:stop] = y_school
    private_component_all[start:stop] = private_component
    epsilon_all[start:stop] = epsilon

    start = stop

df["private_component"] = private_component_all
df["epsilon"] = epsilon_all
df["gpa"] = y_all

true_parameters = {
    **{name: set_gamma[i] for i, name in enumerate(gamma_names)},
    "lambda": set_lambda_true,
    "beta": set_beta_true,
    "delta": set_delta_true,
    "lambda_1": set_lambda_true + set_delta_true - 1.0,
    "lambda_2": 1.0 - set_delta_true,
}

print("Generated observations:", len(df))
print("Mean GPA:", round(df["gpa"].mean(), 4))
print("True parameters")
print(pd.Series(true_parameters))


# %%
# 5. First-stage fitted outcomes

y_level = df["gpa"].to_numpy(float)
X_level = df[X_COLS].to_numpy(float)
school_id = df["school_id"].to_numpy()

school_dummies = pd.get_dummies(
    df["school_id"],
    prefix="school",
    drop_first=True,
    dtype=float,
).to_numpy()

first_stage_design = np.column_stack(
    [
        np.ones(len(df)),
        X_level,
        school_dummies,
    ]
)

first_stage_coef, *_ = np.linalg.lstsq(
    first_stage_design,
    y_level,
    rcond=None,
)

yhat = first_stage_design @ first_stage_coef

if np.any(yhat <= 0):
    raise ValueError("The CES instrument needs positive first-stage fitted values.")

first_stage_residual = y_level - yhat
first_stage_r2 = 1.0 - np.sum(first_stage_residual ** 2) / np.sum(
    (y_level - y_level.mean()) ** 2
)

print("First-stage R2:", round(first_stage_r2, 4))
print("Mean y:", round(y_level.mean(), 4))
print("Mean yhat:", round(yhat.mean(), 4))

# %%

# 5bis. Plot First stage correlation and network

plt.figure(figsize=(6, 5))
plt.scatter(yhat, y_level, alpha=0.5)

min_value = min(yhat.min(), y_level.min())
max_value = max(yhat.max(), y_level.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1,
)

plt.xlabel("First-stage fitted GPA")
plt.ylabel("Actual GPA")
plt.title(f"First-stage fit: actual GPA vs fitted GPA, R2 = {first_stage_r2:.3f}")
plt.show()

# Diagnose first-stage fit by isolation status

plt.figure(figsize=(6, 5))

plt.scatter(
    yhat[~isolated],
    y_level[~isolated],
    alpha=0.5,
    label="Non-isolated",
)

plt.scatter(
    yhat[isolated],
    y_level[isolated],
    alpha=0.7,
    marker="x",
    label="Isolated",
)

min_value = min(yhat.min(), y_level.min())
max_value = max(yhat.max(), y_level.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1,
)

plt.xlabel("First-stage fitted GPA")
plt.ylabel("Actual GPA")
plt.title(f"Actual GPA vs fitted GPA, R2 = {first_stage_r2:.3f}")
plt.legend()
plt.show()

# %%
# 6. Fixed effects and residualized variables

G_block = sparse.block_diag(G_list, format="csr")

row_sum = np.asarray(G_block.sum(axis=1)).ravel()
isolated = row_sum == 0

school_codes, schools = pd.factorize(school_id, sort=True)
demeaning_group = 2 * school_codes + isolated.astype(int)

n_groups = int(demeaning_group.max()) + 1
group_counts = np.bincount(demeaning_group, minlength=n_groups).astype(float)


def residualize_by_group(values):
    """Small mechanical helper: subtract group means."""
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        group_sums = np.bincount(
            demeaning_group,
            weights=values,
            minlength=n_groups,
        )
        group_means = group_sums / group_counts
        return values - group_means[demeaning_group]

    residualized = np.empty_like(values, dtype=float)

    for col in range(values.shape[1]):
        group_sums = np.bincount(
            demeaning_group,
            weights=values[:, col],
            minlength=n_groups,
        )
        group_means = group_sums / group_counts
        residualized[:, col] = values[:, col] - group_means[demeaning_group]

    return residualized


y_residual = residualize_by_group(y_level)
X_residual = residualize_by_group(X_level)

print("Total observations:", len(y_residual))
print("Isolated observations:", int(np.sum(isolated)))
print("Non-isolated observations:", int(np.sum(~isolated)))
print("Block network shape:", G_block.shape)


# %%
# 7. Build the beta-dependent objects once for inspection

beta_for_inspection = set_beta_true

S_level = np.zeros_like(y_level)
peer_sum_y = G_block @ (y_level ** beta_for_inspection)
S_level[~isolated] = peer_sum_y[~isolated] ** (1.0 / beta_for_inspection)

Shat_level = np.zeros_like(yhat)
peer_sum_yhat = G_block @ (yhat ** beta_for_inspection)
Shat_level[~isolated] = peer_sum_yhat[~isolated] ** (1.0 / beta_for_inspection)

yhat_beta = yhat ** beta_for_inspection
Dhat_level = np.zeros_like(yhat)
derivative_sum = G_block @ (yhat_beta * np.log(yhat))
Dhat_level[~isolated] = Shat_level[~isolated] * (
    -np.log(peer_sum_yhat[~isolated]) / (beta_for_inspection ** 2)
    + derivative_sum[~isolated]
    / (beta_for_inspection * peer_sum_yhat[~isolated])
)

S_residual = residualize_by_group(S_level)
Shat_residual = residualize_by_group(Shat_level)
Dhat_residual = residualize_by_group(Dhat_level)

preview = df[["student_id", "school_id", "local_id", *X_COLS, "gpa"]].copy()
preview["isolated"] = isolated
preview["S_level"] = S_level
preview["Shat_level"] = Shat_level
preview["Dhat_level"] = Dhat_level

print(preview.head(8))


# %%
# Plot beta function

beta_grid = np.linspace(0.25, 50, 200)

nonisolated_indices = np.where(~isolated)[0]

students_to_plot = nonisolated_indices[:5]

S_by_beta = np.zeros((len(beta_grid), len(students_to_plot)))

for b_index, beta_value in enumerate(beta_grid):
    peer_sum = G_block @ (y_level ** beta_value)
    S_temp = np.zeros_like(y_level)
    S_temp[~isolated] = peer_sum[~isolated] ** (1.0 / beta_value)

    S_by_beta[b_index, :] = S_temp[students_to_plot]

plt.figure(figsize=(7, 5))

for j, student_index in enumerate(students_to_plot):
    plt.plot(
        beta_grid,
        S_by_beta[:, j],
        label=f"Student {student_index}",
    )

plt.xlabel(r"$\beta$")
plt.ylabel(r"$S_i(\beta)$")
plt.title("Observed peer norm as a function of beta")
plt.legend()
plt.show()


# %%
# Plot observed and fitted peer norm as functions of beta

beta_grid = np.linspace(0.25, 50, 200)

student_to_plot = np.where(~isolated)[0][0]

S_observed_grid = []
S_fitted_grid = []

for beta_value in beta_grid:
    peer_sum_y = G_block @ (y_level ** beta_value)
    S_temp = np.zeros_like(y_level)
    S_temp[~isolated] = peer_sum_y[~isolated] ** (1.0 / beta_value)

    peer_sum_yhat = G_block @ (yhat ** beta_value)
    Shat_temp = np.zeros_like(yhat)
    Shat_temp[~isolated] = peer_sum_yhat[~isolated] ** (1.0 / beta_value)

    S_observed_grid.append(S_temp[student_to_plot])
    S_fitted_grid.append(Shat_temp[student_to_plot])

plt.figure(figsize=(7, 5))

plt.plot(beta_grid, S_observed_grid, label=r"$S_i(\beta)$ using observed GPA")
plt.plot(beta_grid, S_fitted_grid, label=r"$\widehat S_i(\beta)$ using fitted GPA")

plt.xlabel(r"$\beta$")
plt.ylabel("Peer norm")
plt.title(f"Observed and fitted peer norm, student {student_to_plot}")
plt.legend()
plt.show()


# %%
# Plot derivative of fitted peer norm with respect to beta

beta_grid = np.linspace(0.25, 50, 200)

student_to_plot = np.where(~isolated)[0][0]

Dhat_grid = []

for beta_value in beta_grid:
    peer_sum_yhat = G_block @ (yhat ** beta_value)

    Shat_temp = np.zeros_like(yhat)
    Shat_temp[~isolated] = peer_sum_yhat[~isolated] ** (1.0 / beta_value)

    yhat_beta = yhat ** beta_value
    derivative_sum = G_block @ (yhat_beta * np.log(yhat))

    Dhat_temp = np.zeros_like(yhat)
    Dhat_temp[~isolated] = Shat_temp[~isolated] * (
        -np.log(peer_sum_yhat[~isolated]) / (beta_value ** 2)
        + derivative_sum[~isolated]
        / (beta_value * peer_sum_yhat[~isolated])
    )

    Dhat_grid.append(Dhat_temp[student_to_plot])

plt.figure(figsize=(7, 5))

plt.plot(beta_grid, Dhat_grid)

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel(r"$\beta$")
plt.ylabel(r"$\partial \widehat S_i(\beta)/\partial \beta$")
plt.title(f"Derivative of fitted peer norm, student {student_to_plot}")
plt.show()

# %%
# 8. Objective used by scipy.optimize.minimize

def concentrated_gmm_objective(params, W_I_current, W_N_current):
    """
    Concentrated GMM objective.

    scipy.optimize.minimize needs this as a callable function. Inside, we build
    the beta-dependent objects, solve gamma analytically, then evaluate moments.
    """
    lambda_value, beta_value, delta_value = params

    if beta_value <= 0 or delta_value <= 0 or lambda_value < 0:
        return 1e12

    try:
        S_level_current = np.zeros_like(y_level)
        peer_sum_y_current = G_block @ (y_level ** beta_value)

        if np.any(peer_sum_y_current[~isolated] <= 0):
            return 1e12

        S_level_current[~isolated] = (
            peer_sum_y_current[~isolated] ** (1.0 / beta_value)
        )

        Shat_level_current = np.zeros_like(yhat)
        peer_sum_yhat_current = G_block @ (yhat ** beta_value)

        if np.any(peer_sum_yhat_current[~isolated] <= 0):
            return 1e12

        Shat_level_current[~isolated] = (
            peer_sum_yhat_current[~isolated] ** (1.0 / beta_value)
        )

        yhat_beta_current = yhat ** beta_value
        derivative_sum_current = G_block @ (yhat_beta_current * np.log(yhat))

        Dhat_level_current = np.zeros_like(yhat)
        Dhat_level_current[~isolated] = Shat_level_current[~isolated] * (
            -np.log(peer_sum_yhat_current[~isolated]) / (beta_value ** 2)
            + derivative_sum_current[~isolated]
            / (beta_value * peer_sum_yhat_current[~isolated])
        )

        if (
            not np.all(np.isfinite(S_level_current))
            or not np.all(np.isfinite(Shat_level_current))
            or not np.all(np.isfinite(Dhat_level_current))
        ):
            return 1e12

        S_current = residualize_by_group(S_level_current)
        Shat_current = residualize_by_group(Shat_level_current)
        Dhat_current = residualize_by_group(Dhat_level_current)

        X_I = X_residual[isolated]
        y_I = y_residual[isolated]

        X_N = X_residual[~isolated]
        y_N = y_residual[~isolated]
        S_N = S_current[~isolated]
        Shat_N = Shat_current[~isolated]
        Dhat_N = Dhat_current[~isolated]

        Z_N = np.column_stack([X_N, Shat_N, Dhat_N])

        N_I = X_I.shape[0]
        N_N = X_N.shape[0]

        # Here we concentrate out gamma
        
        isolated_gamma_jacobian = X_I.T @ X_I / N_I
        isolated_outcome_moment = X_I.T @ y_I / N_I

        nonisolated_gamma_jacobian = delta_value * Z_N.T @ X_N / N_N
        nonisolated_outcome_moment = Z_N.T @ (
            y_N - lambda_value * S_N
        ) / N_N

        A = (
            isolated_gamma_jacobian.T
            @ W_I_current
            @ isolated_gamma_jacobian
            + nonisolated_gamma_jacobian.T
            @ W_N_current
            @ nonisolated_gamma_jacobian
        )
        b = (
            isolated_gamma_jacobian.T @ W_I_current @ isolated_outcome_moment
            + nonisolated_gamma_jacobian.T
            @ W_N_current
            @ nonisolated_outcome_moment
        )

        gamma_current = np.linalg.solve(A, b)
        # end of concentration
        
        epsilon_I = y_I - X_I @ gamma_current
        e_N = (
            y_N
            - delta_value * (X_N @ gamma_current)
            - lambda_value * S_N
        )

        m_I = X_I.T @ epsilon_I / N_I
        m_N = Z_N.T @ e_N / N_N

        objective = m_I @ W_I_current @ m_I + m_N @ W_N_current @ m_N

        if not np.isfinite(objective):
            return 1e12

        return float(objective)

    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return 1e12


# %%
# 9. First-step GMM with identity weights

W_I_first = np.eye(k)
W_N_first = np.eye(k + 2)

first_step_results = []

for start_value in starting_values:
    result = minimize(
        concentrated_gmm_objective,
        x0=start_value,
        args=(W_I_first, W_N_first),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 300,
            "ftol": 1e-10,
        },
    )
    first_step_results.append(result)

first_step_optimizer = min(first_step_results, key=lambda item: item.fun)
lambda_first, beta_first, delta_first = first_step_optimizer.x

print("First-step nonlinear parameters")
print(pd.Series(
    {
        "lambda": lambda_first,
        "beta": beta_first,
        "delta": delta_first,
        "objective": first_step_optimizer.fun,
    }
))


# %%
# 10. Rebuild objects at the first-step beta and solve gamma

S_level_first = np.zeros_like(y_level)
peer_sum_y_first = G_block @ (y_level ** beta_first)
S_level_first[~isolated] = peer_sum_y_first[~isolated] ** (1.0 / beta_first)

Shat_level_first = np.zeros_like(yhat)
peer_sum_yhat_first = G_block @ (yhat ** beta_first)
Shat_level_first[~isolated] = (
    peer_sum_yhat_first[~isolated] ** (1.0 / beta_first)
)

yhat_beta_first = yhat ** beta_first
derivative_sum_first = G_block @ (yhat_beta_first * np.log(yhat))
Dhat_level_first = np.zeros_like(yhat)
Dhat_level_first[~isolated] = Shat_level_first[~isolated] * (
    -np.log(peer_sum_yhat_first[~isolated]) / (beta_first ** 2)
    + derivative_sum_first[~isolated]
    / (beta_first * peer_sum_yhat_first[~isolated])
)

S_first = residualize_by_group(S_level_first)
Shat_first = residualize_by_group(Shat_level_first)
Dhat_first = residualize_by_group(Dhat_level_first)

X_I = X_residual[isolated]
y_I = y_residual[isolated]

X_N = X_residual[~isolated]
y_N = y_residual[~isolated]
S_N_first = S_first[~isolated]
Shat_N_first = Shat_first[~isolated]
Dhat_N_first = Dhat_first[~isolated]

Z_N_first = np.column_stack([X_N, Shat_N_first, Dhat_N_first])

N_I = X_I.shape[0]
N_N = X_N.shape[0]

isolated_gamma_jacobian = X_I.T @ X_I / N_I
isolated_outcome_moment = X_I.T @ y_I / N_I

nonisolated_gamma_jacobian = delta_first * Z_N_first.T @ X_N / N_N
nonisolated_outcome_moment = Z_N_first.T @ (
    y_N - lambda_first * S_N_first
) / N_N

A_first = (
    isolated_gamma_jacobian.T @ W_I_first @ isolated_gamma_jacobian
    + nonisolated_gamma_jacobian.T @ W_N_first @ nonisolated_gamma_jacobian
)
b_first = (
    isolated_gamma_jacobian.T @ W_I_first @ isolated_outcome_moment
    + nonisolated_gamma_jacobian.T @ W_N_first @ nonisolated_outcome_moment
)

gamma_first = np.linalg.solve(A_first, b_first)

first_step_estimates = {
    **{name: gamma_first[i] for i, name in enumerate(gamma_names)},
    "lambda": lambda_first,
    "beta": beta_first,
    "delta": delta_first,
    "lambda_1": lambda_first + delta_first - 1.0,
    "lambda_2": 1.0 - delta_first,
}

print("First-step parameter estimates")
print(pd.Series(first_step_estimates).round(4))


# %%
# 11. Build second-step weight matrices

epsilon_I_first = y_I - X_I @ gamma_first
e_N_first = (
    y_N
    - delta_first * (X_N @ gamma_first)
    - lambda_first * S_N_first
)

q_I_first = X_I * epsilon_I_first[:, None]
q_N_first = Z_N_first * e_N_first[:, None]

S_I_cov = q_I_first.T @ q_I_first / N_I
S_N_cov = q_N_first.T @ q_N_first / N_N

W_I_second = np.linalg.pinv(S_I_cov)
W_N_second = np.linalg.pinv(S_N_cov)

print("Shape of W_I_second:", W_I_second.shape)
print("Shape of W_N_second:", W_N_second.shape)


# %%
# 12. Second-step GMM with updated weights

second_step_starting_values = [
    first_step_optimizer.x,
    *starting_values,
]

second_step_results = []

for start_value in second_step_starting_values:
    result = minimize(
        concentrated_gmm_objective,
        x0=start_value,
        args=(W_I_second, W_N_second),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 300,
            "ftol": 1e-10,
        },
    )
    second_step_results.append(result)

second_step_optimizer = min(second_step_results, key=lambda item: item.fun)
lambda_second, beta_second, delta_second = second_step_optimizer.x

print("Second-step nonlinear parameters")
print(pd.Series(
    {
        "lambda": lambda_second,
        "beta": beta_second,
        "delta": delta_second,
        "objective": second_step_optimizer.fun,
    }
))


# %%
# 13. Rebuild objects at the second-step beta and solve final gamma

S_level_second = np.zeros_like(y_level)
peer_sum_y_second = G_block @ (y_level ** beta_second)
S_level_second[~isolated] = (
    peer_sum_y_second[~isolated] ** (1.0 / beta_second)
)

Shat_level_second = np.zeros_like(yhat)
peer_sum_yhat_second = G_block @ (yhat ** beta_second)
Shat_level_second[~isolated] = (
    peer_sum_yhat_second[~isolated] ** (1.0 / beta_second)
)

yhat_beta_second = yhat ** beta_second
derivative_sum_second = G_block @ (yhat_beta_second * np.log(yhat))
Dhat_level_second = np.zeros_like(yhat)
Dhat_level_second[~isolated] = Shat_level_second[~isolated] * (
    -np.log(peer_sum_yhat_second[~isolated]) / (beta_second ** 2)
    + derivative_sum_second[~isolated]
    / (beta_second * peer_sum_yhat_second[~isolated])
)

S_second = residualize_by_group(S_level_second)
Shat_second = residualize_by_group(Shat_level_second)
Dhat_second = residualize_by_group(Dhat_level_second)

S_N_second = S_second[~isolated]
Shat_N_second = Shat_second[~isolated]
Dhat_N_second = Dhat_second[~isolated]
Z_N_second = np.column_stack([X_N, Shat_N_second, Dhat_N_second])

nonisolated_gamma_jacobian = delta_second * Z_N_second.T @ X_N / N_N
nonisolated_outcome_moment = Z_N_second.T @ (
    y_N - lambda_second * S_N_second
) / N_N

A_second = (
    isolated_gamma_jacobian.T @ W_I_second @ isolated_gamma_jacobian
    + nonisolated_gamma_jacobian.T @ W_N_second @ nonisolated_gamma_jacobian
)
b_second = (
    isolated_gamma_jacobian.T @ W_I_second @ isolated_outcome_moment
    + nonisolated_gamma_jacobian.T @ W_N_second @ nonisolated_outcome_moment
)

try:
    gamma_second = np.linalg.solve(A_second, b_second)
except np.linalg.LinAlgError:
    gamma_second = np.linalg.pinv(A_second) @ b_second

second_step_estimates = {
    **{name: gamma_second[i] for i, name in enumerate(gamma_names)},
    "lambda": lambda_second,
    "beta": beta_second,
    "delta": delta_second,
    "lambda_1": lambda_second + delta_second - 1.0,
    "lambda_2": 1.0 - delta_second,
}

print("Second-step parameter estimates")
print(pd.Series(second_step_estimates).round(4))


# %%
# 14. Classical non-clustered standard errors

def moments_for_theta(theta):
    """Stacked moments used only for the numerical Jacobian below."""
    gamma_value = theta[:k]
    lambda_value, beta_value, delta_value = theta[k:]

    S_level_current = np.zeros_like(y_level)
    peer_sum_y_current = G_block @ (y_level ** beta_value)
    S_level_current[~isolated] = (
        peer_sum_y_current[~isolated] ** (1.0 / beta_value)
    )

    Shat_level_current = np.zeros_like(yhat)
    peer_sum_yhat_current = G_block @ (yhat ** beta_value)
    Shat_level_current[~isolated] = (
        peer_sum_yhat_current[~isolated] ** (1.0 / beta_value)
    )

    yhat_beta_current = yhat ** beta_value
    derivative_sum_current = G_block @ (yhat_beta_current * np.log(yhat))
    Dhat_level_current = np.zeros_like(yhat)
    Dhat_level_current[~isolated] = Shat_level_current[~isolated] * (
        -np.log(peer_sum_yhat_current[~isolated]) / (beta_value ** 2)
        + derivative_sum_current[~isolated]
        / (beta_value * peer_sum_yhat_current[~isolated])
    )

    S_current = residualize_by_group(S_level_current)
    Shat_current = residualize_by_group(Shat_level_current)
    Dhat_current = residualize_by_group(Dhat_level_current)

    S_N_current = S_current[~isolated]
    Shat_N_current = Shat_current[~isolated]
    Dhat_N_current = Dhat_current[~isolated]
    Z_N_current = np.column_stack([X_N, Shat_N_current, Dhat_N_current])

    epsilon_I_current = y_I - X_I @ gamma_value
    e_N_current = (
        y_N
        - delta_value * (X_N @ gamma_value)
        - lambda_value * S_N_current
    )

    m_I_current = X_I.T @ epsilon_I_current / N_I
    m_N_current = Z_N_current.T @ e_N_current / N_N

    return np.concatenate([m_I_current, m_N_current])


theta_hat = np.concatenate(
    [
        gamma_second,
        np.array([lambda_second, beta_second, delta_second]),
    ]
)

base_moments = moments_for_theta(theta_hat)
jacobian = np.zeros((len(base_moments), len(theta_hat)))

for j, value in enumerate(theta_hat):
    step = max(abs(value) * 1e-5, 1e-5)

    theta_high = theta_hat.copy()
    theta_low = theta_hat.copy()
    theta_high[j] += step
    theta_low[j] -= step

    if parameter_names[j] == "beta" and theta_low[j] <= 0:
        theta_low[j] = value
        theta_high[j] = value + step
        moments_high = moments_for_theta(theta_high)
        jacobian[:, j] = (moments_high - base_moments) / step
    else:
        moments_high = moments_for_theta(theta_high)
        moments_low = moments_for_theta(theta_low)
        jacobian[:, j] = (moments_high - moments_low) / (2.0 * step)

epsilon_all_final = y_residual - X_residual @ gamma_second
e_all_final = (
    y_residual
    - delta_second * (X_residual @ gamma_second)
    - lambda_second * S_second
)

moment_contributions = np.zeros((len(df), 2 * k + 2))
moment_contributions[isolated, :k] = (
    X_residual[isolated] * epsilon_all_final[isolated, None] / N_I
)
moment_contributions[~isolated, k:] = (
    Z_N_second * e_all_final[~isolated, None] / N_N
)

n_parameters = len(parameter_names)
small_sample_correction = len(df) / max(len(df) - n_parameters, 1)
omega = small_sample_correction * (
    moment_contributions.T @ moment_contributions
)

W_block = np.zeros((2 * k + 2, 2 * k + 2))
W_block[:k, :k] = W_I_second
W_block[k:, k:] = W_N_second

bread = jacobian.T @ W_block @ jacobian
middle = jacobian.T @ W_block @ omega @ W_block @ jacobian
bread_inv = np.linalg.pinv(bread)

parameter_covariance = bread_inv @ middle @ bread_inv

transform_rows = []
for i in range(len(parameter_names)):
    row = np.zeros(len(parameter_names))
    row[i] = 1.0
    transform_rows.append(row)

lambda_index = parameter_names.index("lambda")
delta_index = parameter_names.index("delta")

lambda_1_row = np.zeros(len(parameter_names))
lambda_1_row[lambda_index] = 1.0
lambda_1_row[delta_index] = 1.0
transform_rows.append(lambda_1_row)

lambda_2_row = np.zeros(len(parameter_names))
lambda_2_row[delta_index] = -1.0
transform_rows.append(lambda_2_row)

transform = np.vstack(transform_rows)
reported_covariance = transform @ parameter_covariance @ transform.T
classical_se = np.sqrt(np.maximum(np.diag(reported_covariance), 0.0))


# %%
# 15. Final simple parameter table

final_table = pd.DataFrame(
    {
        "true": pd.Series(true_parameters),
        "estimate": pd.Series(second_step_estimates),
        "classical_se": pd.Series(classical_se, index=reported_names),
    }
).reindex(reported_names)

final_table["t_stat"] = final_table["estimate"] / final_table["classical_se"]

comparison_table = pd.DataFrame(
    {
        "true": pd.Series(true_parameters),
        "first_step": pd.Series(first_step_estimates),
        "second_step": pd.Series(second_step_estimates),
    }
).reindex(reported_names)

print()
print("Parameter comparison")
print(comparison_table.round(4))
print()
print("Final estimates with classical non-clustered SE")
print(final_table.round(4))

comparison_table.to_csv(OUTPUT_DIR / "line_by_line_parameter_comparison.csv")
final_table.to_csv(OUTPUT_DIR / "line_by_line_final_estimates.csv")

print()
print("Saved simple outputs to:", OUTPUT_DIR)
