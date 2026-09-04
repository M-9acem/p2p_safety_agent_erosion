"""Network-level adapter drift and ASR summary statistics (spec section 6).

Per-pair drift is average.adapter_drift; this module aggregates it (and
per-agent ASR) across the whole network for the report/plots.
"""
from __future__ import annotations

import statistics
from itertools import combinations
from typing import Any

from p2p_safety.average import Adapter, adapter_drift


def network_drift(adapters: dict[int, Adapter], alpha: float, r: int) -> dict[str, float]:
    """Mean pairwise adapter drift across all agent pairs."""
    ids = sorted(adapters)
    if len(ids) < 2:
        raise ValueError("need at least 2 agents to compute pairwise drift")
    pairwise = [
        adapter_drift(adapters[i], adapters[j], alpha, r) for i, j in combinations(ids, 2)
    ]
    return {
        "mean": statistics.mean(pairwise),
        "min": min(pairwise),
        "max": max(pairwise),
        "variance": statistics.variance(pairwise) if len(pairwise) > 1 else 0.0,
    }


def network_asr_summary(asr_by_agent: dict[int, float]) -> dict[str, float]:
    """Mean/min/max/variance of ASR across agents (spec section 6,
    network-level metrics)."""
    values = list(asr_by_agent.values())
    if not values:
        raise ValueError("asr_by_agent must be non-empty")
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
    }


def asr_by_hop_distance(
    asr_by_agent: dict[int, float], hop_distance_by_agent: dict[int, int]
) -> dict[int, dict[str, Any]]:
    """ASR as a function of hop distance from the nearest safety-holding
    agent — the RQ3 plot (spec section 6). Groups agents by hop distance
    and summarizes ASR within each group."""
    by_hop: dict[int, list[float]] = {}
    for agent_id, asr in asr_by_agent.items():
        hop = hop_distance_by_agent[agent_id]
        by_hop.setdefault(hop, []).append(asr)

    return {
        hop: {
            "mean_asr": statistics.mean(values),
            "n_agents": len(values),
        }
        for hop, values in sorted(by_hop.items())
    }
