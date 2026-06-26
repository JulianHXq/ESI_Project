"""
Line-by-line teaching script for Project Replication Zenou -- v2.

"""

import numpy as np
import pandas as pd

from DGP_v5 import (
    DATA_DIR,
    X_COLS,
    build_environment,
    load_environment,
    save_environment,
    summarize_environment,
)
from Estimation_v5 import (
    OUTPUT_DIR,
    STARTING_VALUES,
    build_estimation_data,
    cluster_robust_standard_errors,
    comparison_table,
    estimate_weight_matrices,
    estimate_with_weights,
    final_estimation_table,
    prepare_estimation_context,
    save_estimation_outputs,
)


# ============================================================
# Small display helpers
# ============================================================

def network_summary(raw_G_list, G_list):
    """Summarize friendship networks school by school."""
    rows = []

    for school_id, (raw_G, G) in enumerate(zip(raw_G_list, G_list)):
        out_degree = np.asarray(raw_G.sum(axis=1)).ravel()
        isolated = np.asarray(G.sum(axis=1)).ravel() == 0

        rows.append(
            {
                "school_id": school_id,
                "n_students": G.shape[0],
                "n_isolated": int(np.sum(isolated)),
                "share_isolated": float(np.mean(isolated)),
                "mean_friends": float(np.mean(out_degree)),
                "mean_friends_nonisolated": float(np.mean(out_degree[~isolated])),
                "max_friends": int(np.max(out_degree)),
            }
        )

    return pd.DataFrame(rows)


def first_stage_summary(df, context):
    """Summarize the first-stage fitted values used to build instruments."""
    y = df["gpa"].to_numpy(float)
    yhat = context["yhat"]
    residual = y - yhat
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)

    return pd.Series(
        {
            "n_observations": len(y),
            "first_stage_r2": r2,
            "mean_y": np.mean(y),
            "mean_yhat": np.mean(yhat),
            "sd_y": np.std(y, ddof=1),
            "sd_yhat": np.std(yhat, ddof=1),
            "min_yhat": np.min(yhat),
            "max_yhat": np.max(yhat),
        }
    )


def context_summary(context):
    """Show dimensions of the prepared estimation objects."""
    G = context["G"]

    return pd.Series(
        {
            "n_observations": len(context["y"]),
            "n_schools": len(context["schools"]),
            "n_isolated": context["n_isolated"],
            "n_nonisolated": context["n_nonisolated"],
            "network_rows": G.shape[0],
            "network_cols": G.shape[1],
            "network_nonzero_links": G.nnz,
        }
    )


def beta_object_summary(beta, beta_data):
    """Summarize peer norms and instruments for one beta."""
    nonisolated = ~beta_data["isolated"]

    S_level = beta_data["S_level"][nonisolated]
    Shat_level = beta_data["Shat_level"][nonisolated]
    Dhat_level = beta_data["Dhat_level"][nonisolated]

    S_residual = beta_data["S"][nonisolated]
    Shat_residual = beta_data["Shat"][nonisolated]
    Dhat_residual = beta_data["Dhat"][nonisolated]

    def clean_decimal(value):
        if abs(value) < 0.00005:
            value = 0.0

        return f"{value:,.4f}"

    rows = [
        {"object": "beta", "value": clean_decimal(beta)},
        {"object": "non-isolated students", "value": f"{int(np.sum(nonisolated)):,}"},
        {"object": "mean peer norm S, levels", "value": clean_decimal(np.mean(S_level))},
        {"object": "sd peer norm S, levels", "value": clean_decimal(np.std(S_level, ddof=1))},
        {
            "object": "mean peer norm S, FE residual",
            "value": clean_decimal(np.mean(S_residual)),
        },
        {
            "object": "sd peer norm S, FE residual",
            "value": clean_decimal(np.std(S_residual, ddof=1)),
        },
        {
            "object": "mean predicted peer norm Shat, levels",
            "value": clean_decimal(np.mean(Shat_level)),
        },
        {
            "object": "sd predicted peer norm Shat, levels",
            "value": clean_decimal(np.std(Shat_level, ddof=1)),
        },
        {
            "object": "mean predicted peer norm Shat, FE residual",
            "value": clean_decimal(np.mean(Shat_residual)),
        },
        {
            "object": "sd predicted peer norm Shat, FE residual",
            "value": clean_decimal(np.std(Shat_residual, ddof=1)),
        },
        {
            "object": "mean derivative Dhat, levels",
            "value": clean_decimal(np.mean(Dhat_level)),
        },
        {
            "object": "sd derivative Dhat, levels",
            "value": clean_decimal(np.std(Dhat_level, ddof=1)),
        },
        {
            "object": "mean derivative Dhat, FE residual",
            "value": clean_decimal(np.mean(Dhat_residual)),
        },
        {
            "object": "sd derivative Dhat, FE residual",
            "value": clean_decimal(np.std(Dhat_residual, ddof=1)),
        },
        {
            "object": "corr(S, Shat), FE residuals",
            "value": clean_decimal(np.corrcoef(S_residual, Shat_residual)[0, 1]),
        },
    ]

    return pd.DataFrame(rows)


def beta_object_preview(df, beta_data, n=10):
    """Show the first n students with beta-dependent peer objects attached."""
    preview = df[["student_id", "school_id", "local_id", *X_COLS, "gpa"]].copy()

    preview["isolated"] = beta_data["isolated"]
    preview["y_FE_residual"] = beta_data["y"]
    preview["S_level"] = beta_data["S_level"]
    preview["S_FE"] = beta_data["S_fixed_effect"]
    preview["S_FE_residual"] = beta_data["S"]
    preview["Shat_level"] = beta_data["Shat_level"]
    preview["Shat_FE_residual"] = beta_data["Shat"]
    preview["Dhat_level"] = beta_data["Dhat_level"]
    preview["Dhat_FE_residual"] = beta_data["Dhat"]

    return preview.head(n)


def weight_matrix_print(W_I, W_N, x_cols=X_COLS):
    """Print labeled GMM weight matrices rounded to two decimals."""

    iso_labels = list(x_cols)
    connected_labels = list(x_cols) + ["Shat", "Dhat"]

    W_I_df = pd.DataFrame(
        W_I,
        index=iso_labels,
        columns=iso_labels,
    )

    W_N_df = pd.DataFrame(
        W_N,
        index=connected_labels,
        columns=connected_labels,
    )

    print("Weight matrix for isolated students")
    print(W_I_df.round(2))
    print()

    print("Weight matrix for connected students")
    print(W_N_df.round(2))
    

def assemble_results(context, first_step, second_step, W_I, W_N):
    """Bundle estimates and weights in the format expected by reporting code."""
    return {
        "first_step": first_step,
        "second_step": second_step,
        "W_I": W_I,
        "W_N": W_N,
        "yhat": context["yhat"],
        "n_observations": int(len(context["y"])),
        "n_clusters": int(len(context["schools"])),
    }


# ============================================================
# Step 1. Generate the synthetic data
# ============================================================

df, raw_G_list, G_list, true_parameters = build_environment()
save_environment(df, raw_G_list, G_list, true_parameters, DATA_DIR)

print(summarize_environment(df, G_list, true_parameters))


# If you do not want to regenerate data, comment out the block above and use:
# df, raw_G_list, G_list, true_parameters = load_environment(DATA_DIR)


# ============================================================
# Step 2. Inspect the generated environment
# ============================================================

true = pd.Series(true_parameters, name="true")
networks = network_summary(raw_G_list, G_list)

print()
print("True parameters")
print(true)

print()
print("Network summary")
print(networks.head())


# ============================================================
# Step 3. Prepare reusable estimation objects
# ============================================================

context = prepare_estimation_context(df, G_list)

print()
print("Prepared context")
print(context_summary(context))

print()
print("First-stage summary")
print(first_stage_summary(df, context))


# ============================================================
# Step 4. Inspect beta-dependent peer objects
# ============================================================

beta_to_inspect = 10
data_beta = build_estimation_data(beta=beta_to_inspect, context=context)

print()
print("Beta-dependent peer objects")
print(beta_object_summary(beta_to_inspect, data_beta))

print()
print("Preview of peer objects")
print(beta_object_preview(df, data_beta, n=8))


# ============================================================
# Step 5. First-step GMM with identity weights
# ============================================================

first_step = estimate_with_weights(
    context=context,
    stage="first_step_identity",
)

print()
print("First-step estimates")
print(pd.Series(first_step["estimates"]))


# ============================================================
# Step 6. Update the GMM weighting matrices
# ============================================================

lambda_1, beta_1, delta_1 = first_step["nonlinear_parameters"]

W_I, W_N = estimate_weight_matrices(
    data=first_step["data"],
    gamma=first_step["gamma"],
    lambda_value=lambda_1,
    delta_value=delta_1,
)

print()
print("Weight matrix summary")
print(weight_matrix_print(W_I, W_N))


# ============================================================
# Step 7. Second-step GMM with updated weights
# ============================================================

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

print()
print("Second-step estimates")
print(pd.Series(second_step["estimates"]))


# ============================================================
# Step 8. Cluster-robust standard errors
# ============================================================

results = assemble_results(context, first_step, second_step, W_I, W_N)
results["cluster_robust_se"] = cluster_robust_standard_errors(
    context=context,
    results=results,
)

comparison = comparison_table(true_parameters, results)
final = final_estimation_table(true_parameters, results)

print()
print("True vs first-step vs second-step")
print(comparison)

print()
print("Final estimates with school-cluster robust SE")
print(final)


# ============================================================
# Step 9. Save outputs
# ============================================================

save_estimation_outputs(results, comparison, OUTPUT_DIR)

print()
print(f"Saved outputs to: {OUTPUT_DIR}")
