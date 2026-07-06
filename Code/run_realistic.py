"""[v5-ext] Realistic-network scenario driver.

    python run_realistic.py

Builds the extended DGP with homophily + triadic closure + selective isolation +
a latent ability, prints the network diagnostics, and shows how the peers-of-peers
estimator behaves -- the overid Hansen J-test DETECTS that the realistic features
break the exclusion restriction. The plain extended DGP (run_assignment.py) is unaffected.
"""
import DGP_v5 as D
import Estimation_v5 as E


def main():
    df, raw, G, tp = D.ext_build_environment_realistic(seed=2026)
    print("Network diagnostics (realistic vs a random network):")
    for k, v in D.ext_network_diagnostics(df, raw).items():
        print(f"  {k}: {v:.3f}")
    ec, _ = E.ext_estimate(df, G, "correct")
    en, _ = E.ext_estimate(df, G, "naive")
    J = E.ext_overid_j_test(df, G)
    print("\nEstimation on realistic data (true lambda=0.30, beta=5.0):")
    print("  correct: lambda=%.3f beta=%.2f | naive: lambda=%.3f beta=%.2f"
          % (ec["lambda"], ec["beta"], en["lambda"], en["beta"]))
    verdict = "REJECTED -> instruments invalid (realistic homophily breaks the exclusion restriction)" \
        if J["reject_5pct"] else "not rejected"
    print("  Hansen overid J=%.2f (p=%.3f): %s" % (J["J"], J["p_value"], verdict))


if __name__ == "__main__":
    main()
