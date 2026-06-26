"""
Run EVERYTHING for v3 in one command:

    python run_all_v5.py

Pipeline:
  1) Teaching walkthrough (two-step internals)        -> teaching_steps_v5.py
  2) Data + CUE estimation (with two-step baseline)    -> rich console report
  3) Figures and slide-style figures                   -> Figures_v5 / SlideFigures_v5

The CUE outputs are saved last among the estimation steps, so the persisted
estimation_results.json (used by the figures) reflects CUE.
(Heavy extras fig10/fig12/fig13 stay manual: run monte_carlo_* / identification_frontier
 and the matching fig_* functions.)
"""

import os
import runpy

from DGP_v5 import DATA_DIR, build_environment, save_environment, summarize_environment
from Estimation_v5 import (
    OUTPUT_DIR,
    comparison_table,
    estimate_model_cue,      # [v3]
    format_full_report,      # [v3]
    compare_instrument_sets, # [v5]
    format_instrument_comparison,  # [v5]
    save_estimation_outputs,
)
import Figures_v5            # [v3] has main()
import SlideFigures_v5       # [v3] has main()

HERE = os.path.dirname(os.path.abspath(__file__))


def _banner(text):
    print("\n" + "#" * 70 + f"\n# {text}\n" + "#" * 70 + "\n")


def main():
    # 1) Teaching walkthrough. teaching_steps_v5 is a flat script, so we run it
    #    with runpy. It saves a two-step JSON that step 2 then overwrites.
    _banner("1/3  TEACHING WALKTHROUGH (two-step internals)")
    runpy.run_path(os.path.join(HERE, "teaching_steps_v5.py"), run_name="__main__")

    # 2) Data + CUE estimation (headline). CUE outputs persist.
    _banner("2/3  FULL ESTIMATION (CUE, two-step baseline)")
    df, raw_G_list, G_list, true_parameters = build_environment()
    save_environment(df, raw_G_list, G_list, true_parameters, DATA_DIR)
    print(summarize_environment(df, G_list, true_parameters))
    print()
    results = estimate_model_cue(df, G_list)
    comparison = comparison_table(true_parameters, results)
    save_estimation_outputs(results, comparison, OUTPUT_DIR)
    print(format_full_report(true_parameters, results))

    # [v5] instrument-transformation comparison (which instruments estimate worse)
    _banner("EXTRA  INSTRUMENT TRANSFORMATIONS")
    _itab = compare_instrument_sets(df, G_list)
    print(format_instrument_comparison(_itab, true_beta=true_parameters["beta"]))

    # 3) Figures (read the saved CUE data + JSON).
    _banner("3/3  FIGURES")
    Figures_v5.main()
    SlideFigures_v5.main()

    print(f"\nAll done. Data: {DATA_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Figures: figures_v5/  (fig1-9, fig11 + slide1-6)")


if __name__ == "__main__":
    main()
