"""Cross-cutting engineering requirements (spec section 9 / 10):
full seeding, and the git SHA every run must log alongside its config.
"""
from __future__ import annotations

import random
import subprocess


def seed_everything(seed: int) -> None:
    """Seed python, numpy (if installed), and torch (if installed), and
    set torch's deterministic flags. Call once at the start of every run.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_sha(short: bool = False) -> str:
    """Current commit SHA, for logging alongside every run's resolved
    config (spec section 10: "a result you cannot trace back to the exact
    code that produced it is not a result").

    Raises if the working tree has uncommitted changes, since a run
    against dirty state can't be traced back to a commit — commit first
    per section 10, then launch the run.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise RuntimeError(
            "working tree has uncommitted changes; commit before launching "
            "a run so its results are traceable to a commit (spec section 10)"
        )
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
