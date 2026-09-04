"""General-capability and task-performance eval.

Spec section 6: MMLU subset (separates targeted safety loss from broad
forgetting) plus the task's held-out split. Needs real model inference
(transformers/vllm), so this is a Phase 1 stub with a fixed interface —
fill in `run_mmli_subset` / `run_held_out_task` once the model-loading
stack is available; simulate.py and analysis/tables.py should not need to
change when that happens.
"""
from __future__ import annotations

from typing import Any, Protocol


class Generator(Protocol):
    """Whatever agent.py exposes for batched generation, e.g. a bound
    method on Agent wrapping HF `generate` or vLLM."""

    def __call__(self, prompts: list[str]) -> list[str]: ...


def run_mmlu_subset(generate: Generator, subset_size: int, seed: int = 0) -> dict[str, Any]:
    raise NotImplementedError(
        "Phase 1: load an MMLU subset, format as multiple-choice prompts, "
        "score exact-match on the answer letter"
    )


def run_held_out_task(generate: Generator, held_out_examples: list[dict]) -> dict[str, Any]:
    raise NotImplementedError(
        "Phase 1: score the task's held-out split (e.g. ROUGE/exact-match "
        "against the reference output, per data.task_dataset)"
    )
