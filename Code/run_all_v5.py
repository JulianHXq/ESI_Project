"""
Run EVERYTHING for v5 in one clean pass:

    python run_all_v5.py

The data are generated ONCE and reused; nothing is printed or estimated twice.
  1) DGP summary                                   (printed once)
  2) Teaching walkthrough of the two-step internals (reuses the data)
  3) CUE estimation + full report                   (the headline results)
  4) Instrument-transformation comparison
  5) Figures and slide-style figures
(Heavy extras fig10/fig12/fig13 and the Monte Carlo / frontier stay manual.)
"""

from DGP_v5 import DATA_DIR, build_environment, save_environment, summarize_environment
from Estimation_v5 import (
    OUTPUT_DIR,
    comparison_table,
    estimate_model_cue,
    format_full_report,
    compare_instrument_sets,
    format_instrument_comparison,
    save_estimation_outputs,
)
import Figures_v5
import SlideFigures_v5
import teaching_steps_v5


def _banner(text):
    print("\n" + "#" * 70 + f"\n# {text}\n" + "#" * 70 + "\n")


def main():
    # ---- 1) Generate the data ONCE and show the summary once ----
    _banner("1/5  SYNTHETIC DATA")
    df, raw_G_list, G_list, true_parameters = build_environment()
    save_environment(df, raw_G_list, G_list, true_parameters, DATA_DIR)
    print(summarize_environment(df, G_list, true_parameters))

    # ---- 2) Teaching walkthrough on the SAME data (no rebuild / no re-summary / no save) ----
    _banner("2/5  TEACHING WALKTHROUGH (two-step internals)")
    teaching_steps_v5.main(env=(df, raw_G_list, G_list, true_parameters),
                           save=False, show_summary=False)

    # ---- 3) CUE estimation + full report (this is what persists to outputs_v5) ----
    _banner("3/5  CUE ESTIMATION + FULL REPORT")
    results = estimate_model_cue(df, G_list)
    comparison = comparison_table(true_parameters, results)
    save_estimation_outputs(results, comparison, OUTPUT_DIR)
    print(format_full_report(true_parameters, results))

    # ---- 4) Instrument-transformation comparison ----
    _banner("4/5  INSTRUMENT TRANSFORMATIONS")
    print(format_instrument_comparison(compare_instrument_sets(df, G_list),
                                       true_beta=true_parameters["beta"]))

    # ---- 5) Figures ----
    _banner("5/5  FIGURES")
    Figures_v5.main()
    SlideFigures_v5.main()

    print(f"\nAll done. Data: {DATA_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Figures: figures_v5/")


if __name__ == "__main__":
    main()
