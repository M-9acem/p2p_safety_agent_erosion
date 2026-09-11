"""Summary tables for the report: per-phase acceptance numbers, the RQ3
headline statement ("k agents, placed how, keep the network within
tolerance x of baseline"), and seed-variance summaries.

Same record shapes as plots.py — see that module's docstring.
"""
from __future__ import annotations

from statistics import mean, pvariance
from typing import Any


def summarize_asr_by_seed(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean/variance of final-round ASR across seeds, per topology (Phase
    3's deliverable — "does connectivity density change it", spec RQ2).
    Each row of `results` needs "topology", "seed", "final_asr". Returns
    {topology: {"mean_final_asr", "seed_variance", "seed_std", "n_seeds"}}."""
    if not results:
        raise ValueError("results is empty")

    by_topology: dict[str, list[float]] = {}
    for row in results:
        by_topology.setdefault(row["topology"], []).append(row["final_asr"])

    summary = {}
    for topology, values in by_topology.items():
        variance = pvariance(values) if len(values) > 1 else 0.0
        summary[topology] = {
            "mean_final_asr": mean(values),
            "seed_variance": variance,
            "seed_std": variance**0.5,
            "n_seeds": len(values),
        }
    return summary


def rq3_headline_table(
    results: list[dict[str, Any]], baseline_asr: float, tolerance: float = 0.05
) -> dict[str, dict[str, Any]]:
    """For each placement strategy, the minimum k that brings network mean
    ASR within `tolerance` (absolute) of `baseline_asr` — the spec's
    Phase 4 acceptance criterion ("a clear quantitative statement of how
    many such agents, placed where, keep the network within tolerance of
    baseline"). Each row of `results` needs "k", "placement", "asr_mean"
    (k=0 rows, if present, are ignored — they're the baseline itself, not
    a placement to rank).

    Returns {placement: {"min_k_within_tolerance": int | None, "asr_at_min_k":
    float | None, "k_values_tried": [...]}}; min_k is None if no k in the
    sweep got within tolerance."""
    if not results:
        raise ValueError("results is empty")

    by_placement: dict[str, list[dict]] = {}
    for row in results:
        if row["k"] == 0:
            continue
        by_placement.setdefault(row["placement"], []).append(row)

    table = {}
    for placement, rows in by_placement.items():
        rows = sorted(rows, key=lambda r: r["k"])
        within = [r for r in rows if abs(r["asr_mean"] - baseline_asr) <= tolerance]
        best = min(within, key=lambda r: r["k"]) if within else None
        table[placement] = {
            "min_k_within_tolerance": best["k"] if best else None,
            "asr_at_min_k": best["asr_mean"] if best else None,
            "k_values_tried": [r["k"] for r in rows],
        }
    return table
