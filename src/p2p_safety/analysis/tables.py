"""Summary tables for the report: per-phase acceptance numbers, the RQ3
headline statement ("k agents, placed how, keep the network within
tolerance x of baseline"), and seed-variance summaries.

Deferred: needs real run data. See plots.py for the same caveat.
"""
from __future__ import annotations

from typing import Any


def summarize_asr_by_seed(results: list[dict[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError("Phase 3: mean/variance of final-round ASR across seeds, per topology")


def rq3_headline_table(results: list[dict[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError(
        "Phase 4: for each (k, placement), the minimum k that keeps network "
        "mean ASR within tolerance of baseline"
    )
