"""Multi-run sweeps (n_agents x graph topology x seed for Phase 3;
k x placement for Phase 4) via Hydra's multirun.

Usage (once deps are installed):
    python scripts/sweep.py --multirun graph=ring,random,complete seed=0,1,2
"""
from __future__ import annotations

from run_experiment import main

if __name__ == "__main__":
    main()
