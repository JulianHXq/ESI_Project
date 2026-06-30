"""
Unified runner for the complete v5 project -- ONE command runs EVERYTHING:

    python run_all_v5.py

PART A -- BASELINE (Boucher, Rendall, Ushchev & Zenou 2024): the class model
          y = delta*p + lambda*S(beta), estimated by two-step GMM + CUE, with
          Anderson-Rubin, LIM, Hansen J and an instrument-set comparison.
          Direct-peer / intransitivity instruments are valid (no contextual effect).

PART B -- EXTENDED MODEL (contextual peer effects): adds a contextual term
          phi'(Gx); direct-peer instruments become invalid, so the norm is
          identified with peers-of-peers (G^2x). Heterogeneous schools, weak-IV F
          test and a 2x2 control/instrument sensitivity. All functions live in the
          v5 modules under the ext_ prefix.

Bridge: PART B's "naive" mode IS PART A's class estimator -- the one that becomes
inconsistent once the contextual effect is present. Both parts share core.py.
"""


def _banner(t):
    print("\n" + "#" * 72 + f"\n# {t}\n" + "#" * 72 + "\n")


def run_baseline():
    _banner("PART A  --  BASELINE (Boucher et al. 2024): the class model")
    from DGP_v5 import (DATA_DIR, build_environment, save_environment,
                        summarize_environment)
    from Estimation_v5 import (OUTPUT_DIR, comparison_table, estimate_model_cue,
                               format_full_report, compare_instrument_sets,
                               format_instrument_comparison, save_estimation_outputs)
    import Figures_v5, SlideFigures_v5, teaching_steps_v5
    df, raw_G_list, G_list, true = build_environment()
    save_environment(df, raw_G_list, G_list, true, DATA_DIR)
    print(summarize_environment(df, G_list, true))
    _banner("A.2  Teaching walkthrough (two-step internals)")
    teaching_steps_v5.main(env=(df, raw_G_list, G_list, true), save=False, show_summary=False)
    _banner("A.3  CUE estimation + full report")
    results = estimate_model_cue(df, G_list)
    save_estimation_outputs(results, comparison_table(true, results), OUTPUT_DIR)
    print(format_full_report(true, results))
    _banner("A.4  Instrument transformations")
    print(format_instrument_comparison(compare_instrument_sets(df, G_list),
                                       true_beta=true["beta"]))
    _banner("A.5  Baseline figures")
    Figures_v5.main(); SlideFigures_v5.main()


def run_extended():
    _banner("PART B  --  EXTENDED MODEL (contextual peer effects, peers-of-peers IV)")
    import json
    import DGP_v5 as D, Estimation_v5 as E, Figures_v5 as F
    df, raw_G_list, G_list, true = D.ext_build_environment()
    D.ext_save_environment(df, raw_G_list, G_list, true)
    print(D.ext_summarize_environment(df, G_list, true)); print()
    table, (ec, sc), (en, sn) = E.ext_compare_estimators(df, G_list, true)
    E.EXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(E.EXT_OUTPUT_DIR / "estimator_comparison.csv")
    with (E.EXT_OUTPUT_DIR / "estimates.json").open("w") as f:
        json.dump({"correct": ec, "correct_se": sc, "naive": en, "naive_se": sn}, f, indent=2)
    print("True vs NAIVE (class, inconsistent) vs CORRECT (extended model)\n")
    print(table.round(4).to_string())
    Fstat = E.ext_first_stage_F(df, G_list, beta=ec["beta"])
    stab = E.ext_sensitivity_table(df, G_list, true)
    print("\nWeak-instrument test (peers-of-peers G^2x for the norm): "
          "partial F = %.1f | partial R2 = %.3f | strong(F>10): %s"
          % (Fstat["F_peers_of_peers"], Fstat["partial_R2"], Fstat["strong"]))
    print("\nSensitivity (contextual control x instrument):")
    print(stab.round(3).to_string())
    stab.to_csv(E.EXT_OUTPUT_DIR / "sensitivity.csv")
    with (E.EXT_OUTPUT_DIR / "weak_iv_test.json").open("w") as f:
        json.dump(Fstat, f, indent=2)
    ctx = E.ext_prepare_context(df, G_list)
    F.ext_fig_ego_network(raw_G_list); F.ext_fig_first_stage(ctx)
    F.ext_fig_objective_profiles(ctx); F.ext_fig_bias(table); F.ext_fig_sensitivity(stab)


def main():
    run_baseline()
    run_extended()
    print("\n" + "=" * 72)
    print("DONE.  Everything under v5: Outputs/v5, Figures/v5, Data/generated_v5 (+ /extended)")
    print("=" * 72)


if __name__ == "__main__":
    main()
