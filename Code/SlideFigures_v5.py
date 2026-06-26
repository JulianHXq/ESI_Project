"""
Slide-style figures for Project Replication Zenou -- v5.

Recreates, with the simulated data, the kinds of figures used in the course
slides (network graphs, the adjacency matrix, the intransitive triad, binned
peer scatters, the outcome distribution, and the social multiplier).

    python DGP_v5.py
    python SlideFigures_v5.py

Figures are written to the `figures/` folder. Titles use Title Case.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy import sparse

from DGP_v5 import (PROJECT_ROOT, load_environment, ces_norm,
                 generate_school_network)

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

FIG = PROJECT_ROOT / "figures_v5"; FIG.mkdir(exist_ok=True)
INK, ACC, BLU, RED, GRY = "#23211D", "#8C7B66", "#33475B", "#B23A48", "#999999"
plt.rcParams.update({"axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.dpi": 150})


def _layout(adj):
    if HAS_NX:
        Gx = nx.from_numpy_array(((adj + adj.T) > 0).astype(float))
        p = nx.spring_layout(Gx, seed=3, k=1.6 / np.sqrt(max(len(Gx), 1)))
        return np.array([p[i] for i in range(adj.shape[0])])
    rng = np.random.default_rng(3); return rng.normal(size=(adj.shape[0], 2))


def _draw_network(ax, adj, node_color=None, cmap="viridis", cbar_label=None, fig=None):
    pos = _layout(adj)
    ei, ej = np.where(adj > 0)
    for s, t in zip(ei, ej):
        ax.plot([pos[s, 0], pos[t, 0]], [pos[s, 1], pos[t, 1]], color="#cfcfcf", lw=0.5, zorder=1)
    if node_color is None:
        ax.scatter(pos[:, 0], pos[:, 1], s=55, color=ACC, edgecolor="white", lw=0.6, zorder=2)
    else:
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=node_color, cmap=cmap, s=70,
                        edgecolor="white", lw=0.6, zorder=2)
        if fig is not None and cbar_label:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.axis("off")


def _solve_eq(G, p, beta, lam, delta, isolated, max_iter=2000, tol=1e-10):
    y = p.copy()
    for _ in range(max_iter):
        S = ces_norm(G, y, beta, isolated)
        yn = delta * p + lam * S
        yn[isolated] = p[isolated]
        if np.max(np.abs(yn - y)) < tol:
            return yn
        y = yn
    return y


# S1 -- Disperse vs Interconnected networks (Angelucci Fig. 4 style)
def fig_disperse_vs_interconnected(n=45):
    rng = np.random.default_rng(7)
    rawD, _ = generate_school_network(n_students=n, max_friends=2, p_isolated=0.45, rng=rng)
    rawC, _ = generate_school_network(n_students=n, max_friends=10, p_isolated=0.03, rng=rng)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for a, raw, name in [(ax[0], rawD.toarray(), "Disperse"), (ax[1], rawC.toarray(), "Interconnected")]:
        deg = raw.sum(1); iso = (deg == 0).mean()
        _draw_network(a, raw)
        a.set_title(f"{name} Network", )
        a.text(0.5, -0.04, f"mean out-degree = {deg.mean():.1f}   ·   isolated = {iso*100:.0f}%",
               transform=a.transAxes, ha="center", fontsize=9, color=INK)
    fig.suptitle("Friendship Networks at Equal Size: Disperse vs Interconnected",
                 fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "slide1_networks_disperse_vs_interconnected.png", bbox_inches="tight")
    plt.close(fig)


# S2 -- the adjacency matrix G (De Giorgi "G-matrix snapshot")
def fig_adjacency_matrix(df, G_list, m=35):
    G = G_list[0].toarray()[:m, :m]
    fig, a = plt.subplots(figsize=(6.4, 5.6))
    im = a.imshow(G, cmap="Greys", vmin=0, vmax=G.max() if G.max() > 0 else 1)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label=r"Edge Weight $g_{ij}$")
    a.set_title("The Adjacency Matrix G (a Snapshot)")
    a.set_xlabel("Named Friend  $j$"); a.set_ylabel("Student  $i$")
    a.text(0.5, -0.17, r"Row-normalized: $\sum_j g_{ij}=1$ for non-isolated students.",
           transform=a.transAxes, ha="center", fontsize=9, color=GRY)  # [v2] moved below
    fig.tight_layout(); fig.savefig(FIG / "slide2_adjacency_matrix.png", bbox_inches="tight")
    plt.close(fig)


# S3 -- the intransitive triad (identification device)
def fig_intransitive_triad():
    fig, a = plt.subplots(figsize=(7.6, 4.6))
    P = {"A": (0.1, 0.2), "B": (0.5, 0.75), "C": (0.9, 0.2)}
    for name, (x, y) in P.items():
        a.add_patch(plt.Circle((x, y), 0.07, color=BLU, zorder=3))
        a.text(x, y, name, color="white", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    def arrow(p, q, color=INK, style="-|>"):
        a.add_patch(FancyArrowPatch(P[p], P[q], arrowstyle=style, mutation_scale=18,
                    color=color, lw=2, shrinkA=16, shrinkB=16, zorder=2))
    arrow("A", "B"); arrow("B", "C")
    a.plot([P["A"][0], P["C"][0]], [P["A"][1], P["C"][1]], ls=":", color=RED, lw=1.6, zorder=1)
    a.text(0.5, 0.12, "no link", color=RED, ha="center", fontsize=9)
    a.text(0.30, 0.52, "A names B", fontsize=9, color=INK)
    a.text(0.70, 0.52, "B names C", fontsize=9, color=INK)
    a.text(0.5, 0.96, r"$C$'s traits affect $A$ only through $B$  $\Rightarrow$  valid instrument",
           ha="center", fontsize=10, color=INK)
    a.set_xlim(0, 1); a.set_ylim(0, 1.05); a.axis("off")
    a.set_title("An Intransitive Triad: The Source of Identification")
    fig.tight_layout(); fig.savefig(FIG / "slide3_intransitive_triad.png", bbox_inches="tight")
    plt.close(fig)


# S4 -- own vs peer-group-mean GPA, binned scatter (Chetty/Sacerdote style)
def fig_binned_peer_scatter(df, G_list, nbins=20):
    G = sparse.block_diag(G_list, format="csr")
    y = df["gpa"].to_numpy(float)
    isolated = np.asarray(G.sum(axis=1)).ravel() == 0
    peer_mean = np.asarray(G @ y).ravel()
    m = ~isolated
    pm, yy = peer_mean[m], y[m]
    qs = np.quantile(pm, np.linspace(0, 1, nbins + 1))
    qs[-1] += 1e-9
    idx = np.digitize(pm, qs[1:-1])
    bx = np.array([pm[idx == b].mean() for b in range(nbins)])
    by = np.array([yy[idx == b].mean() for b in range(nbins)])
    slope, intercept = np.polyfit(pm, yy, 1)
    fig, a = plt.subplots(figsize=(6.8, 5.2))
    a.scatter(bx, by, s=42, color=BLU, zorder=3, label="Binned Means (20 Bins)")
    xs = np.linspace(pm.min(), pm.max(), 50)
    a.plot(xs, intercept + slope * xs, color=RED, lw=2, label=f"OLS Fit (slope = {slope:.2f})")
    a.set_xlabel("Peer-Group Mean GPA"); a.set_ylabel("Own GPA")
    a.set_title("Own GPA vs Peer-Group Mean GPA")
    a.grid(True, alpha=0.18)  # [v2] tidier
    a.legend(fontsize=9, loc="upper left")
    a.text(0.97, 0.04, "Descriptive association, not the structural effect.",
           transform=a.transAxes, ha="right", fontsize=8, color=GRY)
    fig.tight_layout(); fig.savefig(FIG / "slide4_binned_peer_scatter.png", bbox_inches="tight")
    plt.close(fig)


# S5 -- peer effects reshape the grade distribution (mechanism deck spirit)
def fig_distribution_by_intensity(df, G_list, true, n_schools=25):
    beta, delta = true["beta"], true["delta"]
    start = 0; offs = []
    for G in G_list:
        offs.append((start, start + G.shape[0])); start += G.shape[0]
    pall = df["private_component"].to_numpy(float)
    regimes = [("No Peer Effects (λ=0)", 0.0, GRY),
               ("Moderate (λ=0.25)", 0.25, BLU),
               ("Strong (λ=0.50)", 0.50, RED)]
    fig, a = plt.subplots(figsize=(7.6, 4.8))
    for name, lam, col in regimes:
        ys, cvs = [], []
        for G in G_list[:n_schools]:
            n = G.shape[0]
            iso = np.asarray(G.sum(1)).ravel() == 0
            # slice this school's private component
        # build per-school p using offsets
        ys = []
        cv_list = []
        for k, G in enumerate(G_list[:n_schools]):
            lo, hi = offs[k]
            p = pall[lo:hi]; iso = np.asarray(G.sum(1)).ravel() == 0
            yk = _solve_eq(G, p, beta, lam, delta, iso)
            ys.append(yk); cv_list.append(yk.std() / yk.mean())
        yv = np.concatenate(ys)
        a.hist(yv, bins=50, density=True, histtype="step", lw=2, color=col,
               label=f"{name}:  mean={yv.mean():.2f},  CV={np.mean(cv_list):.3f}")
    a.set_title("Peer Effects Reshape the Grade Distribution")
    a.set_xlabel("GPA"); a.set_ylabel("Density")
    a.legend(fontsize=8.5, title="Regime (within-school CV averaged)", title_fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "slide5_distribution_by_intensity.png", bbox_inches="tight")
    plt.close(fig)


# S6 -- the social multiplier: response of mean GPA to a uniform shock
def fig_social_multiplier(df, G_list, true, n_schools=25, dshock=0.20):
    beta, delta = true["beta"], true["delta"]
    start = 0; offs = []
    for G in G_list:
        offs.append((start, start + G.shape[0])); start += G.shape[0]
    pall = df["private_component"].to_numpy(float)
    lambdas = np.linspace(0.0, 0.6, 13)
    emp = []
    for lam in lambdas:
        d_means = []
        for k, G in enumerate(G_list[:n_schools]):
            lo, hi = offs[k]
            p = pall[lo:hi]; iso = np.asarray(G.sum(1)).ravel() == 0
            y0 = _solve_eq(G, p, beta, lam, delta, iso)
            y1 = _solve_eq(G, p + dshock, beta, lam, delta, iso)
            d_means.append((y1 - y0).mean())
        emp.append(np.mean(d_means) / dshock)
    analytic = delta / (1 - lambdas)
    fig, a = plt.subplots(figsize=(7.2, 4.8))
    a.plot(lambdas, emp, "o-", color=BLU, lw=2, label="Simulated  (Δ mean GPA / Δ shock)")
    a.plot(lambdas, analytic, "--", color=RED, lw=1.8, label=r"Analytic  $\delta/(1-\lambda)$")
    a.axhline(delta, color=GRY, ls=":", lw=1, label=r"No Feedback  ($\delta$)")
    a.set_title("The Social Multiplier: Response of Mean GPA to a Uniform Shock")
    a.grid(True, alpha=0.18)  # [v2] tidier
    a.set_xlabel(r"Peer Intensity  $\lambda$"); a.set_ylabel("Amplification Factor")
    a.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "slide6_social_multiplier.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df, raw_G_list, G_list, true = load_environment()
    fig_disperse_vs_interconnected()
    fig_adjacency_matrix(df, G_list)
    fig_intransitive_triad()
    fig_binned_peer_scatter(df, G_list)
    fig_distribution_by_intensity(df, G_list, true)
    fig_social_multiplier(df, G_list, true)
    print(f"Saved 6 slide-style figures to: {FIG}")


if __name__ == "__main__":
    main()
