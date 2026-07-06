"""[v5-ext] Run ONLY the extended (peers-of-peers) model -- the assignment part.

    python run_assignment.py

Generates the extended DGP, runs the correct vs naive estimators, all extensions
(weak-IV F, 2x2 sensitivity, Hansen J, Monte Carlo, network-robust SE, higher-order
G^3x) and the figures. Outputs -> ../Outputs/v5/, figures -> ../Figures/v5/.
"""
from run_all_v5 import run_extended

if __name__ == "__main__":
    run_extended()
