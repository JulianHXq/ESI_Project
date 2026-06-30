# Replicación de Boucher, Rendall, Ushchev & Zenou (2024) — v5 (final)
*Toward a General Theory of Peer Effects*

Versión consolidada y única. v1 (primitivo) y v5 conviven; v2/v3 fueron absorbidos en v5.

## Qué es
Un **Monte Carlo**: generamos datos sintéticos *desde el modelo* con parámetros verdaderos
conocidos, estimamos, y verificamos que los recuperamos. Valida **método + código**, no es
evidencia empírica del mundo real.

## Cómo correr
```
python run_all_v5.py
```
Una sola pasada: datos → walkthrough (two-step) → CUE + reporte → comparación de instrumentos → figuras.
Por partes: `DGP_v5.py` · `Estimation_v5.py` · `Figures_v5.py` · `SlideFigures_v5.py` · `teaching_steps_v5.py`.

## El modelo (una línea)
`y_i = δ·p_i + λ·S_i(β)`, con `S_i(β) = (Σ_j g_ij y_j^β)^(1/β)` la **norma CES**
(β decide *a cuáles* pares se hace caso). Reparametrización: `λ1 = λ+δ−1` (contagio),
`λ2 = 1−δ` (conformismo). Multiplicador social `δ/(1−λ)`.

## Archivos
| archivo | hace | escribe |
|---|---|---|
| `DGP_v5.py` | red + covariables + GPA por punto fijo, clip a [1,4] | `data/generated_v5/` |
| `Estimation_v5.py` | GMM 2-pasos + **CUE**, AR, LIM, test J, F del instrumento, frontera, planner, comparación de instrumentos, reporte | `outputs_v5/` |
| `Figures_v5.py` / `SlideFigures_v5.py` | figuras | `figures_v5/` |
| `teaching_steps_v5.py` | walkthrough paso a paso (función reutilizable) | — |
| `run_all_v5.py` | orquestador de una sola pasada | — |

## Qué cambió vs el primitivo (tags en el código)
- **DGP:** streams RNG independientes `[v2-D]`; GPA en [1,4] (resample + clip del equilibrio) `[v1]/[v2-E]`; diagnósticos del punto fijo.
- **Estimación:** Anderson–Rubin robusto a instrumentos débiles `[v2-A]`; prescan+multistart `[v2-B]`; SE cluster G/(G−1) `[v2-C]`; test LIM `[v2-F]`; **test J de Hansen** `[v5]`; fuerza del instrumento; Monte Carlo; frontera de identificación; planner/key-players; **CUE** `[v3]`; **instrumento configurable + comparación de transformaciones** `[v5]`.
- **Consola:** reporte de 9 secciones (puntos · SE cluster · CIs 95% · test J · F · AR · LIM · optimizador) + tabla de instrumentos.

## Resultados clave (150 escuelas, 30 000 alumnos)
- γ, λ, δ recuperados casi exactos (t de 33 a 93). **β = 9.35** (verdadero 10), SE 1.1; **AR = [9.0, 10.5]** contiene 10.
- **LIM (β=1) rechazado** (objetivo ~39× mayor) → la influencia no es el promedio simple.
- **CUE vs two-step:** medianas casi iguales; CUE tiene **cola más liviana** (menos blow-ups de β), no menor sesgo típico.
- **Instrumentos:** sin D̂, β **no se identifica** (SE 57); D̂² y Ŝ² no ayudan; G²X comparable. `D̂ = ∂Ŝ/∂β` es lo que identifica β.

## Nota honesta
β es el parámetro difícil: a β alto la norma CES se satura y su instrumento se debilita. A 150
escuelas se identifica bien; en muestras chicas se dispersa. Todo es un DGP sintético bien
especificado, por eso la estimación sale limpia.
