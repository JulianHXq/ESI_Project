# Peer Effects with a CES Social Norm

A simulation study that replicates the peer effects model of Boucher, Rendall, Ushchev and Zenou (2024), *Toward a General Theory of Peer Effects*, and extends it to the case where a peer's characteristics affect an individual directly.

The model is estimated on synthetic data drawn from a known data generating process, so every parameter has a true value. The code recovers those values and shows where a naive estimator fails. The exercise therefore validates the method and its implementation; it is not an empirical claim about real data.

## The model

Each individual chooses an outcome (here a GPA) that responds to a CES aggregate of the outcomes of the people they are connected to:

```
y_i = δ · p_i + λ · S_i(β),      S_i(β) = ( Σ_j g_ij · y_j^β )^(1/β)
```

Here `p_i` is the individual's private component, `λ` is the strength of the peer effect, and `β` sets which peers matter: `β = 1` is the simple mean, while a larger `β` puts more weight on the highest achievers. Because each outcome depends on the others, the equilibrium is a fixed point and the peer norm is endogenous, so the model is estimated by GMM.

## Two settings

**Baseline.** A peer's characteristics influence an individual only through that peer's outcome. The endogenous norm is instrumented with functions of peers' characteristics, following Bramoullé, Djebbari and Fortin (2009).

**Extension: direct effects of peers' characteristics.** A contextual term is added so that a peer's characteristics also enter an individual's outcome directly. This invalidates any instrument built from direct peers, so the norm is identified instead with the characteristics of peers of peers, that is, individuals at network distance two. Those characteristics still move the norm but are excluded from the individual's own equation.

A robustness section then stresses the extension with realistic network features (homophily, clustering, selective isolation, and a latent unobserved trait). These break the identifying assumption, and the overidentification test detects the failure.

## Results

- Baseline: the true value `β = 10` is recovered at 9.35, the weak instrument robust confidence set for `β` is [9.0, 10.5], and the restriction that the norm is a simple mean is rejected.
- Extension: ignoring the direct effect of peers' characteristics biases the peer effect upward (`λ = 0.92` against a true 0.30); the estimator based on peers of peers recovers it (`λ = 0.30`, `β = 5.0`) with a very strong first stage (`F ≈ 1.3 × 10⁴`).
- Robustness: under realistic homophily the overidentification test rejects (`p < 0.001`), correctly flagging that the exclusion restriction no longer holds.

## Running it

```bash
cd Code
python run_all_v5.py      # the full study: baseline, extension, and robustness
python run_extension.py  # the extension only
python run_realistic.py   # the robustness stress test only
```

Requirements: numpy, pandas, scipy, matplotlib. The seeds are fixed, so the results are reproducible. Tables are written to `Outputs/` and figures to `Figures/`.

## Repository

- `Code/`
  - `core.py`: the shared CES kernel.
  - `DGP_v5.py`: the data generating processes (baseline, contextual extension, and the realistic network variant).
  - `Estimation_v5.py`: the GMM estimators and every check (weak instruments, control and instrument sensitivity, Hansen J, Monte Carlo, cluster and network robust standard errors, higher order instruments).
  - `Figures_v5.py`, `SlideFigures_v5.py`, `teaching_steps_v5.py`: figures and a step by step walkthrough of the estimator.
  - `run_all_v5.py`, `run_extension.py`, `run_realistic.py`: one command drivers.
- `Data/`, `Outputs/`, `Figures/`: generated artifacts.
- `Notes/`: a written note with the derivations and results (PDF and LaTeX source).
- `Guides/`: reference material.

## References

- Boucher, Rendall, Ushchev and Zenou (2024). *Toward a General Theory of Peer Effects*. Econometrica.
- Bramoullé, Djebbari and Fortin (2009). *Identification of Peer Effects through Social Networks*. Journal of Econometrics.
