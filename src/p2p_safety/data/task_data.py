"""Load the benign task dataset (Alpaca or Dolly) for local SFT.

Training data only. Never import this module from eval/safety.py or vice
versa — see safety_data.py's module docstring for why that boundary is
enforced with a test, not just a convention.
"""
from __future__ import annotations

from typing import Any

_HF_PATHS = {
    "alpaca": "tatsu-lab/alpaca",
    "dolly": "databricks/databricks-dolly-15k",
}

# Tiny offline fallback so partitioning/plumbing can be exercised without
# the `datasets` package or network access. Not representative training
# data — Phase 1 must run against the real HF dataset.
_TINY_FALLBACK = [
    {"instruction": "Summarize the following paragraph.", "input": "...", "output": "..."},
    {"instruction": "List three benefits of exercise.", "input": "", "output": "..."},
    {"instruction": "Translate 'good morning' to French.", "input": "", "output": "Bonjour."},
]


def load_task_dataset(name: str, split: str = "train") -> list[dict[str, Any]]:
    """Load the named benign instruction dataset as a list of examples.

    Requires the `datasets` package; raises ImportError with guidance if
    it isn't installed rather than silently degrading to the tiny fallback,
    since the tiny fallback is not a valid substitute for a real run.
    """
    if name not in _HF_PATHS:
        raise ValueError(f"unknown task dataset: {name}, expected one of {list(_HF_PATHS)}")
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "the `datasets` package is required to load real task data "
            "(pip install datasets); use load_tiny_fallback() only for "
            "offline plumbing tests"
        ) from e

    ds = datasets.load_dataset(_HF_PATHS[name], split=split)
    return list(ds)


def load_tiny_fallback() -> list[dict[str, Any]]:
    """Small in-repo dataset for testing partitioning without network
    access or the `datasets` package. Never used for real experiments."""
    return list(_TINY_FALLBACK)
