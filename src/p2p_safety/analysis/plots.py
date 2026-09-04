"""Plots for the report (spec section 8's Phase 3/4 deliverables):

- ASR-vs-round curves grouped by graph density (Phase 3)
- network ASR vs. k by placement, and ASR vs. hop distance from the
  nearest safety-holding agent — the headline RQ3 figure (Phase 4)

Deferred: needs matplotlib and real run data (W&B history or logged
eval_log from simulate.run_simulation). Add matplotlib to the environment
before implementing.
"""
from __future__ import annotations

from typing import Any


def plot_asr_by_round(eval_log: list[dict[str, Any]], group_by: str, out_path: str) -> None:
    raise NotImplementedError("Phase 3: matplotlib line plot, one line per graph topology")


def plot_asr_vs_k_by_placement(results: list[dict[str, Any]], out_path: str) -> None:
    raise NotImplementedError("Phase 4: the headline figure — network ASR vs k, one line per placement")


def plot_asr_vs_hop_distance(asr_by_hop: dict[int, dict[str, Any]], out_path: str) -> None:
    raise NotImplementedError("Phase 4: ASR vs hop distance from nearest safety-holding agent")
