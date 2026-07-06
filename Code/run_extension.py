"""Run only the extended model (Part B): the contextual peer effects and the
peer-of-peers estimator with all its checks (weak instruments, sensitivity,
Hansen J, Monte Carlo, network robust SE, higher order instruments).

    python run_extension.py
"""
from run_all_v5 import run_extended

if __name__ == "__main__":
    run_extended()
