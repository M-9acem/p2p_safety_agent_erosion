"""Spec section 3 / 9: "the safety eval loader is import-isolated from the
training pipeline, with a test asserting zero overlap between training
corpora and eval prompts."

Two checks:
  1. static: no module on the training path imports safety_data
  2. data: zero overlap between task-data and safety-eval prompt strings
"""
from __future__ import annotations

import ast
from pathlib import Path

from p2p_safety.data.safety_data import load_synthetic_placeholder
from p2p_safety.data.task_data import load_tiny_fallback

SRC = Path(__file__).resolve().parents[1] / "src" / "p2p_safety"

TRAINING_PATH_MODULES = [
    SRC / "agent.py",
    SRC / "simulate.py",
    SRC / "data" / "task_data.py",
    SRC / "data" / "partition.py",
    SRC / "data" / "safety_holding_data.py",
]


def _imported_module_names(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_training_path_never_imports_safety_data():
    for py_file in TRAINING_PATH_MODULES:
        imported = _imported_module_names(py_file)
        offending = {n for n in imported if "safety_data" in n}
        assert not offending, f"{py_file} imports safety_data-related module(s): {offending}"


def test_zero_overlap_between_training_and_eval_prompts():
    task_examples = load_tiny_fallback()
    eval_examples = load_synthetic_placeholder()

    task_strings = {ex["instruction"] for ex in task_examples} | {
        ex["output"] for ex in task_examples
    }
    eval_strings = {ex["prompt"] for ex in eval_examples}

    assert task_strings.isdisjoint(eval_strings)


def _fake_advbench_response(n_rows: int):
    """A fake urlopen() context manager returning a small synthetic
    AdvBench-shaped CSV — numbered rows so overlap is easy to assert on,
    no network needed."""
    from unittest.mock import MagicMock

    lines = ["goal,target"] + [f"goal_{i},target_{i}" for i in range(n_rows)]
    csv_text = "\n".join(lines).encode("utf-8")

    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = csv_text
    return cm


def test_safety_holding_examples_never_overlap_eval_prompts():
    """Real disjointness check (spec section 3/9), not just the two
    reserve-size literals matching: eval takes rows[:-40], safety-holding
    takes rows[-40:], of the SAME underlying source — assert those are
    actually disjoint sets of prompt strings, with each module's own
    (independent, unshared) fetch mocked so this runs offline."""
    from unittest.mock import patch

    from p2p_safety.data import safety_data, safety_holding_data

    with patch("urllib.request.urlopen", return_value=_fake_advbench_response(60)):
        eval_examples = safety_data._load_advbench_csv()
        holding_examples = safety_holding_data.load_safety_holding_examples()

    assert len(eval_examples) == 60 - safety_data.SAFETY_HOLDING_RESERVE_SIZE
    assert len(holding_examples) == safety_data.SAFETY_HOLDING_RESERVE_SIZE

    eval_prompts = {ex["prompt"] for ex in eval_examples}
    holding_prompts = {ex["instruction"] for ex in holding_examples}
    assert eval_prompts.isdisjoint(holding_prompts)
