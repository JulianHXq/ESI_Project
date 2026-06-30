# The Economics of Social Interactions — proyecto v5 (final)

Réplica de **Boucher, Rendall, Ushchev & Zenou (2024)** con su **extensión de efectos
contextuales** (instrumentos de pares-de-pares). **Todo vive bajo v5**: el modelo extendido está
fusionado dentro de los módulos v5 con el prefijo `ext_` (no hay módulos ni carpetas aparte).

## Estructura
- `Code/`     todo el código (7 módulos)
- `Data/`     `generated_v5/` (baseline) con `extended/` adentro (modelo extendido)
- `Outputs/`  `v5/` (tablas de baseline y extendido)
- `Figures/`  `v5/` (figuras de baseline y extendido)
- `Notes/`    notas en PDF + fuentes `.tex` + `RESUMEN_v5.md`
- `Guides/`   material de referencia (paper, assignment, ZenouReplicationNote)

## Correr todo
```bash
cd Code && python run_all_v5.py
```
Una sola pasada corre la Parte A (baseline) y la Parte B (extendido); regenera `Data/`, `Outputs/`, `Figures/` bajo `v5`.

## Las dos partes y el puente
- **Parte A — Baseline (modelo de la clase).** `y = δ·p + λ·S(β)`, norma CES. GMM 2-pasos + CUE,
  Anderson–Rubin, LIM, test J, comparación de instrumentos. Instrumentos de pares directos válidos
  (no hay efecto contextual). β verdadero = 10.
- **Parte B — Modelo extendido (efectos contextuales).** Añade el término contextual `φ'(Gx)`: los
  instrumentos de pares directos se invalidan y la norma se identifica con **pares-de-pares `G²x`**.
  Colegios heterogéneos, test F de instrumentos débiles, sensibilidad 2×2. β verdadero = 5.
  Su código está fusionado en `DGP_v5`/`Estimation_v5`/`Figures_v5` con el prefijo `ext_`.
- **Puente:** el modo `naive` del modelo extendido **es** el estimador de la clase de la Parte A —
  el que se vuelve inconsistente al aparecer el efecto contextual. Las dos partes comparten `core.py`.

## Código (`Code/`)
- `core.py` — kernel CES compartido (`ces_norm`, derivada, `peer_average`)
- `DGP_v5.py` — datos: baseline (`build_environment`) + extendido (`ext_build_environment`)
- `Estimation_v5.py` — baseline (GMM/CUE/AR/LIM/J) + extendido (`ext_estimate`, `ext_first_stage_F`, `ext_sensitivity_table`)
- `Figures_v5.py` — figuras baseline + extendidas (`ext_fig_*`)
- `SlideFigures_v5.py`, `teaching_steps_v5.py` — figuras de diapositiva y walkthrough (baseline)
- `run_all_v5.py` — orquestador (corre baseline y extendido)

## Resultados clave
- Baseline: β = 9.35 (verd. 10), AR = [9.0, 10.5], LIM(β=1) rechazado.
- Extendido: el modo `naive` (clase) sesga (λ=0.92, β=1.97); el modo `correct` recupera
  (λ=0.298, β=4.98), F(G²x) ≈ 1.3×10⁴; la sensibilidad 2×2 muestra que se necesitan control + instrumento.

## Nota
Monte Carlo sintético bien especificado (valida método + código, no es evidencia empírica).
Marcas en el código: `[v5-unify]` (kernel), `[v5-org]` (rutas), `[v5-ext]` (modelo extendido fusionado).
