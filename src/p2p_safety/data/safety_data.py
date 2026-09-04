"""EVAL ONLY. Harmful-behavior prompts for measuring refusal rate.

Spec section 3: "Harmful prompts are for evaluation only, never training."
This module must never be imported from the training path (agent.py's
local-train step, task_data.py, or anything under scripts/run_experiment.py
that builds a training batch). tests/test_data_isolation.py asserts:

  1. this module is not imported anywhere reachable from the training path
  2. zero string overlap between eval prompts and any agent's training shard

Do not add a convenience re-export of these prompts from task_data.py or
agent.py — that would defeat the isolation the test is checking for.
"""
from __future__ import annotations

from typing import Any

# Canonical AdvBench source (Zou et al. 2023), public, no auth required.
# The HF mirror (walledai/AdvBench) is gated (returns 401 on the actual
# parquet without an accepted-terms token) — use the original CSV instead
# so Phase 0 doesn't need an HF token.
_ADVBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

_HF_PATHS = {
    # secondary benchmark, to check the RQ1 result isn't benchmark-specific
    # (spec section 5, Phase 5). Gated on HF — needs `huggingface-cli login`
    # or HF_TOKEN with the dataset's terms accepted before this works.
    "sorry-bench": "sorry-bench/sorry-bench-202406",
}

# Synthetic placeholders shaped like the real schema (a "prompt" field),
# containing no actual harmful content, for exercising the isolation and
# scoring plumbing offline. Never used as real eval data.
_SYNTHETIC_PLACEHOLDER_PROMPTS = [
    {"prompt": "SYNTHETIC_PLACEHOLDER_PROMPT_1"},
    {"prompt": "SYNTHETIC_PLACEHOLDER_PROMPT_2"},
    {"prompt": "SYNTHETIC_PLACEHOLDER_PROMPT_3"},
]


def load_safety_eval(name: str = "advbench") -> list[dict[str, Any]]:
    """Load the named harmful-behaviors benchmark for evaluation only.

    "advbench" fetches the canonical CSV directly (no auth). Anything else
    goes through `datasets.load_dataset` against _HF_PATHS and may need
    `huggingface-cli login` first if that dataset is gated.
    """
    if name == "advbench":
        return _load_advbench_csv()
    if name not in _HF_PATHS:
        raise ValueError(f"unknown safety benchmark: {name}, expected 'advbench' or one of {list(_HF_PATHS)}")
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "the `datasets` package is required to load the real safety "
            "benchmark (pip install datasets); use "
            "load_synthetic_placeholder() only for offline plumbing tests"
        ) from e

    ds = datasets.load_dataset(_HF_PATHS[name], split="train")
    return list(ds)


def _load_advbench_csv() -> list[dict[str, Any]]:
    import csv
    import urllib.request

    with urllib.request.urlopen(_ADVBENCH_CSV_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    # columns are "goal" and "target"; standardize to "prompt" for callers
    return [{"prompt": row["goal"], "target": row["target"]} for row in reader]


def load_synthetic_placeholder() -> list[dict[str, Any]]:
    """Non-harmful placeholder prompts for testing isolation/scoring logic
    without network access or real benchmark data."""
    return list(_SYNTHETIC_PLACEHOLDER_PROMPTS)
