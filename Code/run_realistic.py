"""[v5-ext] Run ONLY the realistic-network stress test (Part C).

    python run_realistic.py

Builds the extended DGP with homophily + triadic closure + selective isolation +
a latent ability, prints the network diagnostics, and shows the peers-of-peers
estimator break -- the overid Hansen J-test detects the invalid exclusion restriction.
"""
from run_all_v5 import run_realistic

if __name__ == "__main__":
    run_realistic()
