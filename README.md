# The Economics of Social Interactions

Réplica de **Boucher, Rendall, Ushchev y Zenou (2024)**, *"Toward a General Theory of Peer Effects"*, junto con una extensión que identifica las interacciones sociales usando **instrumentos de pares de pares** (la tarea del curso; Campa y De Giorgi, Uniandes 2026).

El proyecto tiene tres partes que comparten un mismo kernel CES (`Code/core.py`).

## Las tres partes

**Parte A. Réplica (el modelo de la clase).** Cada estudiante responde a una norma social CES de los resultados de sus pares, `y = δ·p + λ·S(β)`. Se estima con un GMM concentrado en dos pasos (con CUE), más un conjunto de confianza de Anderson y Rubin, un test LIM, un test J de Hansen y una comparación de instrumentos. Aquí las características de los pares directos son instrumentos válidos porque no hay efecto contextual. β verdadero = 10.

**Parte B. Extensión (efectos contextuales).** Las características de los pares directos entran en el resultado de forma directa mediante un término `φ'(Gx)`. Esto invalida los instrumentos de pares directos, así que la norma endógena se identifica con las características de los **pares de pares**, `G²x`. La extensión añade colegios heterogéneos, un test F de instrumentos débiles, una sensibilidad 2×2, un test J, un Monte Carlo, errores estándar robustos a la red e instrumentos de orden mayor. Su código vive en los mismos módulos con el prefijo `ext_`. β verdadero = 5.

**Parte C. Redes realistas (robustez).** Un ejercicio de estrés añade homofilia, cierre triádico, aislamiento selectivo y una habilidad latente. Estas fricciones rompen el supuesto identificador; el test J de Hansen lo detecta incluso cuando las estimaciones puntuales parecen razonables.

El puente entre A y B: el modo `naive` del modelo extendido es exactamente el estimador de la clase, el que se vuelve inconsistente cuando aparece el efecto contextual.

## Cómo correrlo
```bash
cd Code
python run_all_v5.py      # todo: Parte A + Parte B + Parte C
python run_assignment.py  # solo la Parte B (la tarea)
python run_realistic.py   # solo la Parte C (el estrés de robustez)
```
Requisitos: numpy, pandas, scipy, matplotlib. Semillas fijas. Los resultados quedan en `Outputs/` y las figuras en `Figures/`.

## Estructura
- `Code/`: todo el código (ver abajo)
- `Data/`: datos sintéticos generados
- `Outputs/`: tablas y resultados de estimación
- `Figures/`: figuras de diagnóstico y de diapositiva
- `Notes/`: las notas en PDF y sus fuentes LaTeX
- `Guides/`: material de referencia (el paper, la tarea)

## Código
- `core.py`: kernel CES compartido (`ces_norm`, su derivada, `peer_average`).
- `DGP_v5.py`: generación de datos del modelo de la clase, de la extensión contextual (`ext_build_environment`) y de la variante con redes realistas (`ext_build_environment_realistic`, `ext_network_diagnostics`).
- `Estimation_v5.py`: estimación. El GMM/CUE de la clase, y el estimador extendido de pares de pares con todos sus chequeos (`ext_estimate`, `ext_first_stage_F`, `ext_sensitivity_table`, `ext_overid_j_test`, `ext_monte_carlo`, `ext_network_robust_se`, `ext_higher_order_relevance`).
- `Figures_v5.py`, `SlideFigures_v5.py`, `teaching_steps_v5.py`: figuras y un recorrido paso a paso del estimador.
- `run_all_v5.py`, `run_assignment.py`, `run_realistic.py`: orquestadores de un solo comando.

## Resultados principales
- Réplica: β recuperado en 9.35 (verdadero 10), conjunto de Anderson y Rubin [9.0, 10.5], y se rechaza la restricción de la media (β = 1).
- Extensión: el estimador de la clase sobreestima el efecto de pares (λ = 0.92, β = 1.97); el de pares de pares lo recupera (λ = 0.298, β = 4.98), con un primer estadio muy fuerte (F ≈ 1.3×10⁴).
- Robustez: bajo homofilia realista el test J de Hansen rechaza (p < 0.001), y así detecta que el supuesto de exclusión se rompe.

## Nota
Todo corre sobre datos sintéticos bien especificados (un Monte Carlo), así que el ejercicio valida el método y el código; no es una afirmación empírica sobre estudiantes reales. Los cambios respecto al código de la clase están marcados en línea con comentarios `[v5-ext]`.
