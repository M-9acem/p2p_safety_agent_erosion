"""Plots for the report (spec section 8's Phase 3/4 deliverables):

- ASR-vs-round curves grouped by graph density (Phase 3)
- network ASR vs. k by placement, and ASR vs. hop distance from the
  nearest safety-holding agent — the headline RQ3 figure (Phase 4)

Every function takes plain records (list[dict] / dict), the same shape the
notebook already parses run logs and round*_asr.json files into — no
dependency on a live run, a DataFrame, or a particular file layout. See
notebooks/results_analysis.ipynb for the parsing side (phase3_df /
trajectory_df construction, and round*_asr.json's "hop_asr" field, which
plot_asr_vs_hop_distance consumes directly with no reshaping).
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

# Same validated categorical palette as the notebook (first 3 slots for
# topology). Fixed assignment, never cycled — see dataviz color-formula.
TOPOLOGY_COLOR = {
    "ring": "#2a78d6",
    "random": "#eb6834",
    "complete": "#1baf7a",
}
# 4th/5th/6th slots for RQ3's placement categories — same palette family.
PLACEMENT_COLOR = {
    "well_connected": "#2a78d6",
    "peripheral": "#eb6834",
    "clustered": "#1baf7a",
    "spread": "#9b59d0",
}
PLACEMENT_ORDER = ["well_connected", "peripheral", "clustered", "spread"]


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e8e7e1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_asr_by_round(eval_log: list[dict[str, Any]], group_by: str, out_path: str) -> None:
    """One line per distinct value of `group_by` (e.g. "topology"), ASR (%)
    vs. round. `eval_log` rows need at least "round", "asr_mean", and the
    `group_by` field — e.g. trajectory_rows from the phase 3 log parser."""
    if not eval_log:
        raise ValueError("eval_log is empty")

    groups: dict[Any, list[dict]] = {}
    for row in eval_log:
        groups.setdefault(row[group_by], []).append(row)

    fig, ax = plt.subplots(figsize=(6, 4))
    for key in sorted(groups, key=str):
        rows = sorted(groups[key], key=lambda r: r["round"])
        color = TOPOLOGY_COLOR.get(key, None)
        ax.plot(
            [r["round"] for r in rows],
            [r["asr_mean"] * 100 for r in rows],
            label=str(key),
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
    ax.set_xlabel("round")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, title=group_by)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_asr_vs_k_by_placement(results: list[dict[str, Any]], out_path: str) -> None:
    """The headline figure: network mean ASR (%) vs. k, one line per
    placement strategy. Each row of `results` needs "k", "placement",
    "asr_mean" (final-round network mean ASR for that config). A row with
    placement=None and k=0 (the no-safety-holding baseline) is drawn as a
    single reference point/line across every panel, not its own series —
    every placement's k=1 point should be compared against it."""
    if not results:
        raise ValueError("results is empty")

    baseline_rows = [r for r in results if r["k"] == 0]
    baseline_asr = baseline_rows[0]["asr_mean"] * 100 if baseline_rows else None

    by_placement: dict[str, list[dict]] = {}
    for row in results:
        if row["k"] == 0:
            continue
        by_placement.setdefault(row["placement"], []).append(row)

    fig, ax = plt.subplots(figsize=(6, 4))
    if baseline_asr is not None:
        ax.axhline(baseline_asr, color="#52514e", linewidth=1.4, linestyle="--", label="baseline (k=0)")
    for placement in PLACEMENT_ORDER:
        if placement not in by_placement:
            continue
        rows = sorted(by_placement[placement], key=lambda r: r["k"])
        ax.plot(
            [r["k"] for r in rows],
            [r["asr_mean"] * 100 for r in rows],
            label=placement,
            color=PLACEMENT_COLOR[placement],
            linewidth=2,
            marker="o",
            markersize=6,
        )
    ax.set_xlabel("k (number of safety-holding agents)")
    ax.set_ylabel("final network mean ASR (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted({r["k"] for r in results if r["k"] != 0}))
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_asr_vs_hop_distance(asr_by_hop: dict[int, dict[str, Any]], out_path: str) -> None:
    """ASR (%) vs. hop distance from the nearest safety-holding agent.
    `asr_by_hop` is exactly eval.drift.asr_by_hop_distance's return value
    (and round*_asr.json's "hop_asr" field) — {hop: {"mean_asr", "n_agents"}}
    — no reshaping needed between a live run and this plot."""
    if not asr_by_hop:
        raise ValueError("asr_by_hop is empty")

    hops = sorted(asr_by_hop, key=int)
    asr_pct = [asr_by_hop[h]["mean_asr"] * 100 for h in hops]
    n_agents = [asr_by_hop[h]["n_agents"] for h in hops]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(hops, asr_pct, color=TOPOLOGY_COLOR["random"], linewidth=2, marker="o", markersize=7)
    for h, a, n in zip(hops, asr_pct, n_agents):
        ax.annotate(f"n={n}", (h, a), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center", color="#52514e")
    ax.set_xlabel("hop distance from nearest safety-holding agent")
    ax.set_ylabel("mean ASR (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(hops)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
