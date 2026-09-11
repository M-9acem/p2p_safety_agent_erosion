"""plots.py / tables.py: same record shapes the notebook already parses run
logs into (see notebooks/results_analysis.ipynb) — these tests build small
synthetic versions of those records rather than depending on a real run."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to render to a file

from p2p_safety.analysis.plots import (
    plot_asr_by_round,
    plot_asr_vs_hop_distance,
    plot_asr_vs_k_by_placement,
)
from p2p_safety.analysis.tables import rq3_headline_table, summarize_asr_by_seed


def test_plot_asr_by_round_writes_a_file(tmp_path):
    eval_log = [
        {"round": 0, "asr_mean": 0.004, "topology": "ring"},
        {"round": 1, "asr_mean": 0.10, "topology": "ring"},
        {"round": 0, "asr_mean": 0.004, "topology": "complete"},
        {"round": 1, "asr_mean": 0.20, "topology": "complete"},
    ]
    out = tmp_path / "asr_by_round.png"
    plot_asr_by_round(eval_log, group_by="topology", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_asr_vs_k_by_placement_writes_a_file(tmp_path):
    results = [
        {"k": 0, "placement": None, "asr_mean": 0.674},
        {"k": 1, "placement": "well_connected", "asr_mean": 0.50},
        {"k": 2, "placement": "well_connected", "asr_mean": 0.30},
        {"k": 4, "placement": "well_connected", "asr_mean": 0.10},
        {"k": 1, "placement": "peripheral", "asr_mean": 0.65},
    ]
    out = tmp_path / "asr_vs_k.png"
    plot_asr_vs_k_by_placement(results, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_asr_vs_hop_distance_writes_a_file(tmp_path):
    asr_by_hop = {
        0: {"mean_asr": 0.05, "n_agents": 1},
        1: {"mean_asr": 0.30, "n_agents": 5},
        2: {"mean_asr": 0.60, "n_agents": 10},
    }
    out = tmp_path / "asr_vs_hop.png"
    plot_asr_vs_hop_distance(asr_by_hop, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_summarize_asr_by_seed_computes_mean_and_variance():
    results = [
        {"topology": "ring", "seed": 0, "final_asr": 0.60},
        {"topology": "ring", "seed": 1, "final_asr": 0.62},
        {"topology": "ring", "seed": 2, "final_asr": 0.64},
        {"topology": "complete", "seed": 0, "final_asr": 0.70},
    ]
    summary = summarize_asr_by_seed(results)
    assert summary["ring"]["n_seeds"] == 3
    assert abs(summary["ring"]["mean_final_asr"] - 0.62) < 1e-9
    assert summary["ring"]["seed_variance"] > 0
    assert summary["complete"]["n_seeds"] == 1
    assert summary["complete"]["seed_variance"] == 0.0


def test_rq3_headline_table_finds_minimum_k_within_tolerance():
    baseline = 0.004  # phase 0's baseline ASR
    results = [
        {"k": 1, "placement": "well_connected", "asr_mean": 0.40},
        {"k": 2, "placement": "well_connected", "asr_mean": 0.20},
        {"k": 4, "placement": "well_connected", "asr_mean": 0.02},  # within 0.05 of baseline
        {"k": 1, "placement": "peripheral", "asr_mean": 0.60},
        {"k": 2, "placement": "peripheral", "asr_mean": 0.55},
        {"k": 4, "placement": "peripheral", "asr_mean": 0.50},  # never gets close
    ]
    table = rq3_headline_table(results, baseline_asr=baseline, tolerance=0.05)

    assert table["well_connected"]["min_k_within_tolerance"] == 4
    assert abs(table["well_connected"]["asr_at_min_k"] - 0.02) < 1e-9
    assert table["peripheral"]["min_k_within_tolerance"] is None
    assert table["peripheral"]["asr_at_min_k"] is None
    assert table["peripheral"]["k_values_tried"] == [1, 2, 4]


def test_rq3_headline_table_ignores_k0_rows():
    results = [
        {"k": 0, "placement": None, "asr_mean": 0.674},
        {"k": 1, "placement": "spread", "asr_mean": 0.01},
    ]
    table = rq3_headline_table(results, baseline_asr=0.004, tolerance=0.05)
    assert "spread" in table
    assert None not in table  # the k=0/baseline row never becomes its own "placement" entry
