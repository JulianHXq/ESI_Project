"""
Figures for Project Replication Zenou -- v2.

Run this after generating data and (optionally) estimating:

    python DGP.py
    python Estimation.py
    python Figures.py

Produces diagnostic and presentation figures in the `figures/` folder.
Plot titles follow Title Case (connectors stay lowercase).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DGP_v5 import PROJECT_ROOT, X_COLS, load_environment, ces_norm
import Estimation_v5 as E

# [v2] np.trapz is deprecated in numpy>=2.0; use np.trapezoid when available
_trapezoid = getattr(np, "trapezoid", np.trapz)

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

FIG_DIR = PROJECT_ROOT / "figures_v5"
FIG_DIR.mkdir(exist_ok=True)

# palette
INK, ACC, BLU, RED, GRY = "#23211D", "#8C7B66", "#33475B", "#B23A48", "#999999"
plt.rcParams.update({"axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.dpi": 150})


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _spring_layout(adj_dense, iters=80, seed=1):
    """Minimal Fruchterman-Reingold layout (fallback if networkx is absent)."""
    rng = np.random.default_rng(seed)
    n = adj_dense.shape[0]
    A = ((adj_dense + adj_dense.T) > 0).astype(float)
    pos = rng.normal(scale=0.5, size=(n, 2))
    k = 1.0 / np.sqrt(n)
    for it in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
        rep = ((k * k / dist)[:, :, None] * (d / dist[:, :, None])).sum(1)
        att = ((dist / k)[:, :, None] * (d / dist[:, :, None]) * A[:, :, None]).sum(1)
        disp = rep - att
        length = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        cap = 0.1 * (1 - it / iters) + 0.01
        pos += (disp / length[:, None]) * np.minimum(length, cap)[:, None]
    return pos


def _layout(adj_dense):
    if HAS_NX:
        Gx = nx.from_numpy_array(((adj_dense + adj_dense.T) > 0).astype(float))
        p = nx.spring_layout(Gx, seed=1, k=1.5 / np.sqrt(max(len(Gx), 1)))
        return np.array([p[i] for i in range(adj_dense.shape[0])])
    return _spring_layout(adj_dense)


def _solve_equilibrium(G, p, beta, lam, delta, isolated, max_iter=2000, tol=1e-10):
    y = p.copy()
    for _ in range(max_iter):
        S = ces_norm(G, y, beta, isolated)
        y_new = delta * p + lam * S
        y_new[isolated] = p[isolated]
        if np.max(np.abs(y_new - y)) < tol:
            y = y_new
            break
        y = y_new
    return y


def _bfs_ball(adj_dense, start, max_nodes=70):
    """Induced subgraph: BFS ball (undirected) around `start`."""
    A = (adj_dense + adj_dense.T) > 0
    seen, frontier = [start], [start]
    while frontier and len(seen) < max_nodes:
        nxt = []
        for u in frontier:
            for v in np.where(A[u])[0]:
                if v not in seen:
                    seen.append(int(v))
                    nxt.append(int(v))
                    if len(seen) >= max_nodes:
                        break
            if len(seen) >= max_nodes:
                break
        frontier = nxt
    return np.array(sorted(seen))


# ------------------------------------------------------------------
# 1. GPA distribution
# ------------------------------------------------------------------
def fig_gpa_distribution(df):
    fig, a = plt.subplots(figsize=(7, 4.2))
    a.hist(df["gpa"], bins=45, color=ACC, edgecolor="white", alpha=0.9)
    for x in (1, 4):
        a.axvline(x, color=BLU, ls="--", lw=1.2)
    a.set_title("Distribution of the Generated GPA")
    a.set_xlabel("GPA"); a.set_ylabel("Frequency")
    a.text(0.02, 0.95, f"min = {df.gpa.min():.2f}   max = {df.gpa.max():.2f}\n"
           f"mean = {df.gpa.mean():.2f}", transform=a.transAxes, va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_gpa_distribution.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 2. Parameter recovery
# ------------------------------------------------------------------
def fig_parameter_recovery(true, res):
    est, se = res["second_step"], res["cluster_robust_se"]["reported_se"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2),
                           gridspec_kw={"width_ratios": [3, 1]})
    names = ["gamma_age", "gamma_female", "gamma_f_col", "lambda", "delta", "lambda_1", "lambda_2"]
    lbl = [r"$\gamma_{age}$", r"$\gamma_{fem}$", r"$\gamma_{fcol}$", r"$\lambda$",
           r"$\delta$", r"$\lambda_1$", r"$\lambda_2$"]
    yp = np.arange(len(names))[::-1]
    ax[0].errorbar([est[n] for n in names], yp, xerr=[1.96 * se[n] for n in names],
                   fmt="o", color=BLU, capsize=3, label="Estimate ±1.96·SE")
    ax[0].scatter([true[n] for n in names], yp, marker="x", s=70, color=RED, zorder=5, label="True")
    ax[0].set_yticks(yp); ax[0].set_yticklabels(lbl); ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title("Parameter Recovery under Two-Step GMM"); ax[0].set_xlabel("Value")
    ax[1].errorbar([est["beta"]], [0], xerr=[1.96 * se["beta"]], fmt="o", color=BLU, capsize=4)
    ax[1].scatter([true["beta"]], [0], marker="x", s=90, color=RED, zorder=5)
    ax[1].scatter([res["first_step"]["beta"]], [0], marker="s", s=45, color=GRY)
    ax[1].set_yticks([]); ax[1].set_xlim(0, 14); ax[1].set_xlabel(r"$\beta$")
    ax[1].set_title(r"Recovery of $\beta$")
    ax[1].text(0.5, 0.78, "■ first step ≈2\n× true =10", transform=ax[1].transAxes, fontsize=8, ha="center")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_parameter_recovery.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 3. GMM objective profile in beta
# ------------------------------------------------------------------
def fig_beta_objective(df, G_list, res):
    ctx = E.prepare_estimation_context(df, G_list)
    f = res["first_step"]; s = res["second_step"]
    data1 = E.build_estimation_data(beta=f["beta"], context=ctx)
    g1 = E.solve_gamma_given_nonlinear_parameters(data1, f["lambda"], f["delta"])
    W_I, W_N = E.estimate_weight_matrices(data1, g1, f["lambda"], f["delta"])
    betas = np.linspace(1.3, 15, 28)
    Q_id = [E.concentrated_gmm_objective([f["lambda"], b, f["delta"]], ctx) for b in betas]
    Q_op = [E.concentrated_gmm_objective([s["lambda"], b, s["delta"]], ctx, W_I, W_N) for b in betas]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, (Q, ttl, bmin) in zip(ax, [(Q_id, "First Step (Identity Weights)", f["beta"]),
                                      (Q_op, "Second Step (Optimal Weights)", s["beta"])]):
        a.plot(betas, Q, color=BLU, lw=2)
        a.axvline(10, color=RED, ls="--", lw=1.2, label="True β = 10")
        a.axvline(bmin, color=ACC, ls=":", lw=1.6, label=f"Minimum ≈ {bmin:.1f}")
        a.set_yscale("log"); a.set_xlabel(r"$\beta$"); a.set_ylabel("GMM Objective (log)")
        a.grid(True, which="both", alpha=0.18)  # [v2] tidier
        a.set_title(ttl); a.legend(fontsize=8)
    fig.suptitle(r"GMM Objective Profile in $\beta$", fontsize=13, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_beta_objective.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 4. CES weights
# ------------------------------------------------------------------
def fig_ces_weights():
    peers = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    betas = np.linspace(0.2, 20, 140)
    W = np.array([peers ** b / (peers ** b).sum() for b in betas])
    fig, a = plt.subplots(figsize=(7, 4.4))
    for k, yv in enumerate(peers):
        a.plot(betas, W[:, k], lw=1.9, label=f"Peer y = {yv}")
    a.axvline(1, color=GRY, ls=":", lw=1); a.text(1.2, 0.02, "β = 1 (Mean)", fontsize=8, color=GRY)
    a.set_title("CES Weight on Each Peer as β Varies")
    a.set_xlabel(r"$\beta$"); a.set_ylabel("Normalized Weight"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_ces_weights.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 5. Network graph + social multiplier  (rescued from Angelucci/De Giorgi)
# ------------------------------------------------------------------
def fig_network_multiplier(df, raw_G_list, true, school=0, shock=1.5):
    raw = raw_G_list[school].toarray()
    dfx = df[df["school_id"] == school].reset_index(drop=True)
    p_full = dfx["private_component"].to_numpy(float)

    indeg = raw.sum(axis=0)
    seed_node = int(np.argmax(indeg))
    nodes = _bfs_ball(raw, seed_node, max_nodes=70)

    sub_raw = raw[np.ix_(nodes, nodes)]
    rs = sub_raw.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):  # [v2] isolated rows -> masked
        inv = np.where(rs > 0, 1.0 / rs, 0.0)
    from scipy import sparse
    G_sub = sparse.csr_matrix(sub_raw * inv[:, None])
    isolated = rs == 0
    p = p_full[nodes]

    lam, delta, beta = true["lambda"], true["delta"], true["beta"]
    y_base = _solve_equilibrium(G_sub, p, beta, lam, delta, isolated)
    k_local = int(np.argmax(sub_raw.sum(axis=0)))      # most-named node in subgraph
    p_shock = p.copy(); p_shock[k_local] += shock
    y_shock = _solve_equilibrium(G_sub, p_shock, beta, lam, delta, isolated)
    dy = y_shock - y_base

    pos = _layout(sub_raw)
    ei, ej = np.where(sub_raw > 0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for a, color, ttl, cmap, cb in [
        (ax[0], y_base, "Network Colored by GPA", "viridis", "GPA"),
        (ax[1], dy, "Ripple of a Single Shock (ΔGPA)", "magma", "ΔGPA")]:
        for s, t in zip(ei, ej):
            a.plot([pos[s, 0], pos[t, 0]], [pos[s, 1], pos[t, 1]],
                   color="#cccccc", lw=0.4, zorder=1)
        sc = a.scatter(pos[:, 0], pos[:, 1], c=color, cmap=cmap, s=70,
                       edgecolor="white", linewidth=0.6, zorder=2)
        a.scatter(pos[k_local, 0], pos[k_local, 1], s=240, facecolors="none",
                  edgecolors=RED, linewidths=2.0, zorder=3)
        a.set_title(ttl); a.axis("off")
        fig.colorbar(sc, ax=a, fraction=0.046, pad=0.04, label=cb)
    fig.suptitle("A School Friendship Network and the Social Multiplier",
                 fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_network_multiplier.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 6. Who is the relevant peer:  LIM (beta=1) vs CES (beta=10)
# ------------------------------------------------------------------
def fig_relevant_peer(df, G_list, true):
    G = E.sparse.block_diag(G_list, format="csr")
    y = df["gpa"].to_numpy(float)
    isolated = np.asarray(G.sum(axis=1)).ravel() == 0
    S_lim = ces_norm(G, y, 1.0, isolated)
    S_ces = ces_norm(G, y, true["beta"], isolated)
    m = ~isolated
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(m)[0], size=min(3000, m.sum()), replace=False)
    fig, a = plt.subplots(figsize=(6.8, 5.2))
    a.scatter(S_lim[idx], S_ces[idx], s=8, alpha=0.25, color=BLU)
    lo, hi = 1, 4
    a.plot([lo, hi], [lo, hi], color=RED, lw=1.5, ls="--", label="45° (Mean = CES)")
    a.set_xlabel(r"Social Norm under LIM, $\beta=1$ (Peer Mean)")
    a.set_ylabel(r"Social Norm under CES, $\beta=10$")
    a.set_title("The Relevant Peer: Mean (β=1) vs CES (β=10)")
    a.legend(fontsize=9)
    a.text(0.03, 0.92, "Points above the line: the relevant\nnorm is the high achievers, not the mean.",
           transform=a.transAxes, fontsize=8, va="top")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig6_relevant_peer.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 7. Key players: concentration of spillover influence (Boucher policy)
# ------------------------------------------------------------------
def fig_key_players(df, G_list, true):
    G = E.sparse.block_diag(G_list, format="csr")
    y = df["gpa"].to_numpy(float)
    isolated = np.asarray(G.sum(axis=1)).ravel() == 0
    beta = true["beta"]
    yb = y ** beta
    A = np.asarray(G @ yb).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):  # [v2] zero-denominator -> masked
        invA = np.where(A > 0, 1.0 / A, 0.0)
    influence = yb * np.asarray(G.T @ invA).ravel()   # total weight each student carries
    infl = np.sort(influence[~isolated])
    cum = np.cumsum(infl) / infl.sum()
    xx = np.arange(1, len(infl) + 1) / len(infl)
    gini = 1 - 2 * _trapezoid(cum, xx)  # [v2] trapezoid
    top10 = 1 - np.interp(0.90, xx, cum)
    fig, a = plt.subplots(figsize=(6.6, 5.2))
    a.plot(xx, cum, color=BLU, lw=2.2, label="Lorenz Curve of Influence")
    a.plot([0, 1], [0, 1], color=GRY, ls="--", lw=1, label="Perfect Equality")
    a.fill_between(xx, cum, xx, color=ACC, alpha=0.25)
    a.set_xlabel("Cumulative Share of Students (Sorted)")
    a.set_ylabel("Cumulative Share of Spillover Influence")
    a.set_title("Key Players: Concentration of Spillover Influence")
    a.text(0.03, 0.95, f"Gini = {gini:.2f}\nTop 10% carry {100*top10:.0f}% of influence",
           transform=a.transAxes, va="top", fontsize=9)
    a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.84))
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig7_key_players.png", bbox_inches="tight")
    plt.close(fig)


# [v2] -------------------------------------------------------------------
def fig_anderson_rubin_beta(res, true):
    """Weak-IV-robust confidence set for beta (J-statistic vs the chi^2 line)."""
    ar = res["anderson_rubin_beta"]
    betas = np.array(ar["betas"], float)
    J = np.array(ar["J"], float)
    crit = ar["critical_value"]
    bhat = res["second_step"]["beta"]
    fig, a = plt.subplots(figsize=(7.6, 4.7))
    a.plot(betas, J, color=BLU, lw=2.2, label=r"$J(\beta)=n\cdot Q(\beta)$")
    a.axhline(crit, color=GRY, ls="--", lw=1.2, label=rf"$\chi^2$ 95% = {crit:.2f}")
    if ar["ci_low"] is not None:
        a.axvspan(ar["ci_low"], ar["ci_high"], color=ACC, alpha=0.30,
                  label=f"AR 95% set = [{ar['ci_low']:.1f}, {ar['ci_high']:.1f}]")
    a.axvline(true["beta"], color=RED, ls="--", lw=1.5, label=rf"True $\beta$ = {true['beta']:.0f}")
    a.axvline(bhat, color=INK, ls=":", lw=1.5, label=rf"Point $\hat\beta$ = {bhat:.2f}")
    a.set_yscale("log"); a.set_xlabel(r"$\beta$"); a.set_ylabel("GMM J-statistic (log)")
    a.set_title("Anderson-Rubin Confidence Set for β (Weak-IV Robust)")
    a.grid(True, which="both", alpha=0.18)  # [v2] tidier
    a.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.92)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig8_anderson_rubin_beta.png", bbox_inches="tight")
    plt.close(fig)


def fig_lim_vs_free(res):
    """Nonlinearity test: GMM objective under LIM (beta=1) vs the free-beta fit."""
    free = res["second_step"]["objective"]
    lim = res["lim_comparison"]["objective"]
    fig, a = plt.subplots(figsize=(5.8, 4.6))
    bars = a.bar(["LIM (β = 1)", "General (β free)"], [lim, free],
                 color=[RED, BLU], width=0.6, edgecolor="white")
    a.set_yscale("log"); a.set_ylabel("GMM Objective (log)")
    a.set_title("Nonlinearity Test: LIM vs the General Model")
    for b, v in zip(bars, [lim, free]):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:.1e}", ha="center", va="bottom", fontsize=9)
    a.set_ylim(top=a.get_ylim()[1] * 4)  # [v2] headroom so the note clears the bar
    a.text(0.97, 0.95, f"LIM fits ~{lim/free:.0f}× worse → β ≠ 1",
           transform=a.transAxes, ha="right", va="top", fontsize=9.5, color=INK,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRY, alpha=0.9))
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig9_lim_vs_free.png", bbox_inches="tight")
    plt.close(fig)


def fig_monte_carlo(mc_df):
    """Sampling performance from Estimation_v5.monte_carlo_v5(): bias and 95%
    coverage per parameter. Pass the returned DataFrame (params in rows)."""
    params = ["lambda", "beta", "delta", "lambda_1", "lambda_2",
              "gamma_age", "gamma_female", "gamma_f_col"]
    params = [p for p in params if p in mc_df.index]
    bias = mc_df.loc[params, "bias"].astype(float)
    cov = mc_df.loc[params, "coverage95"].astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].barh(params[::-1], bias[::-1], color=ACC, edgecolor="white")
    ax[0].axvline(0, color=INK, lw=1); ax[0].set_title("Monte Carlo Bias of Estimates")
    ax[0].set_xlabel("Mean Estimate − True")
    ax[1].barh(params[::-1], cov[::-1], color=BLU, edgecolor="white")
    ax[1].axvline(0.95, color=RED, ls="--", lw=1.4, label="Nominal 95%")
    ax[1].set_xlim(0, 1); ax[1].set_title("95% CI Coverage"); ax[1].set_xlabel("Coverage")
    ax[1].legend(fontsize=8)
    fig.suptitle("Monte Carlo Performance (Independent DGP Draws)",
                 fontsize=13, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig10_monte_carlo.png", bbox_inches="tight")
    plt.close(fig)


# [v2-F] -----------------------------------------------------------------
def fig_policy_counterfactual(df, G_list, res, school=0):
    """Planner's marginal subsidy value per student under the general CES model
    vs LIM (beta=1): the general model targets high-GPA key players; LIM is
    diffuse. Uses the estimated lambda, delta, beta."""
    est = res["second_step"]
    lam, delta, beta = est["lambda"], est["delta"], est["beta"]
    G = G_list[school]
    dfx = df[df["school_id"] == school]
    y = dfx["gpa"].to_numpy(float)
    isolated = np.asarray(G.sum(axis=1)).ravel() == 0
    v_ces = E.planner_marginal_value(G, y, lam, delta, beta, isolated)
    v_lim = E.planner_marginal_value(G, y, lam, delta, 1.0, isolated)
    s_ces, s_lim = v_ces / v_ces.sum(), v_lim / v_lim.sum()

    def lorenz(v):
        x = np.sort(v); c = np.cumsum(x) / x.sum()
        xx = np.arange(1, len(x) + 1) / len(x)
        return xx, c, 1 - 2 * _trapezoid(c, xx)  # [v2] trapezoid

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    # [v2] mean subsidy share per GPA decile -> the slope difference is the story
    nb = 10
    edges = np.quantile(y, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-9
    bid = np.clip(np.digitize(y, edges[1:-1]), 0, nb - 1)
    gx = np.array([y[bid == b].mean() for b in range(nb)])
    cces = np.array([s_ces[bid == b].mean() for b in range(nb)])
    clim = np.array([s_lim[bid == b].mean() for b in range(nb)])
    ax[0].scatter(y, s_ces, s=9, color=BLU, alpha=0.12)
    ax[0].plot(gx, clim, "s-", color=GRY, lw=2.2, label="LIM (β = 1)")
    ax[0].plot(gx, cces, "o-", color=BLU, lw=2.2, label=f"General (β = {beta:.1f})")
    ax[0].set_xlabel("Student GPA (decile mean)"); ax[0].set_ylabel("Mean Optimal Subsidy Share")
    ax[0].set_title("Who the Planner Subsidizes"); ax[0].legend(fontsize=8)
    for v, col, lab in [(v_lim, GRY, "LIM (β = 1)"), (v_ces, BLU, f"General (β = {beta:.1f})")]:
        xx, c, g = lorenz(v)
        ax[1].plot(xx, c, color=col, lw=2.2, label=f"{lab}: Gini = {g:.2f}")
    ax[1].plot([0, 1], [0, 1], color=RED, ls="--", lw=1, label="Perfect Equality")
    ax[1].set_xlabel("Cumulative Share of Students"); ax[1].set_ylabel("Cumulative Subsidy")
    ax[1].set_title("Concentration of Optimal Subsidies"); ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Counterfactual Policy: LIM Subsidizes Diffusely, the General Model Targets Key Players",
                 fontsize=12, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig11_policy_counterfactual.png", bbox_inches="tight")
    plt.close(fig)


def fig_identification_frontier(frontier_df):
    """Precision of beta vs sample size (and instrument strength), from
    Estimation_v5.identification_frontier()."""
    d = frontier_df.sort_values("n_schools")
    fig, a = plt.subplots(figsize=(7.6, 4.8))
    a.plot(d["n_schools"], d["mean_SE_beta"], "o-", color=BLU, lw=2, label="Mean SE(β)")
    a.plot(d["n_schools"], d["rmse_beta"], "s--", color=ACC, lw=1.8, label="RMSE(β)")
    a.set_xlabel("Number of Schools (Clusters)"); a.set_ylabel("SE(β) / RMSE(β)")
    a.set_title("Identification Frontier: Precision of β vs Sample Size")
    a2 = a.twinx()
    a2.plot(d["n_schools"], d["mean_F"], "^:", color=RED, lw=1.6, label="Mean Instrument F")
    a2.set_ylabel("Mean Instrument F (β)", color=RED); a2.tick_params(axis="y", labelcolor=RED)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig12_identification_frontier.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df, raw_G_list, G_list, true = load_environment()
    res = json.load(open(PROJECT_ROOT / "outputs_v5" / "estimation_results.json"))

    fig_gpa_distribution(df)
    fig_parameter_recovery(true, res)
    fig_beta_objective(df, G_list, res)
    fig_ces_weights()
    fig_network_multiplier(df, raw_G_list, true)
    fig_relevant_peer(df, G_list, true)
    fig_key_players(df, G_list, true)

    # [v2] new diagnostic figures (read AR / LIM from the v2 outputs JSON)
    if res.get("anderson_rubin_beta"):
        fig_anderson_rubin_beta(res, true)
    if res.get("lim_comparison"):
        fig_lim_vs_free(res)
    fig_policy_counterfactual(df, G_list, res)   # [v2-F] LIM vs general targeting
    print(f"Saved figures to: {FIG_DIR}  (run Estimation_v5.monte_carlo_v5 -> "
          "fig_monte_carlo, and identification_frontier -> fig_identification_frontier)")


if __name__ == "__main__":
    main()


def fig_cue_vs_two_step(mc_df):  # [v3]
    """Two-step vs CUE: bias and 95% coverage per parameter, side by side.
    Pass the DataFrame from Estimation_v5.monte_carlo_cue_vs_two_step()."""
    order = ["lambda", "beta", "delta", "lambda_1", "lambda_2",
             "gamma_age", "gamma_female", "gamma_f_col"]
    params = [p for p in order if (p, "cue") in mc_df.index]
    y = np.arange(len(params)); h = 0.38
    bias_ts = [float(mc_df.loc[(p, "two_step"), "bias"]) for p in params]
    bias_cue = [float(mc_df.loc[(p, "cue"), "bias"]) for p in params]
    cov_ts = [float(mc_df.loc[(p, "two_step"), "coverage95"]) for p in params]
    cov_cue = [float(mc_df.loc[(p, "cue"), "coverage95"]) for p in params]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].barh(y + h / 2, bias_ts[::1], height=h, color=ACC, edgecolor="white", label="Two-step")
    ax[0].barh(y - h / 2, bias_cue[::1], height=h, color=BLU, edgecolor="white", label="CUE")
    ax[0].axvline(0, color=INK, lw=1)
    ax[0].set_yticks(y); ax[0].set_yticklabels(params)
    ax[0].set_title("Monte Carlo Bias of Estimates")
    ax[0].set_xlabel("Mean Estimate − True"); ax[0].legend(fontsize=8)
    ax[1].barh(y + h / 2, cov_ts, height=h, color=ACC, edgecolor="white")
    ax[1].barh(y - h / 2, cov_cue, height=h, color=BLU, edgecolor="white")
    ax[1].axvline(0.95, color=RED, ls="--", lw=1.4, label="Nominal 95%")
    ax[1].set_xlim(0, 1); ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_title("95% CI Coverage"); ax[1].set_xlabel("Coverage"); ax[1].legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.suptitle("Two-step vs CUE (Independent DGP Draws)",
                 fontsize=13, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig13_cue_vs_two_step.png", bbox_inches="tight")
    plt.close(fig)
