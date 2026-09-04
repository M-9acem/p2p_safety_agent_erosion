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
