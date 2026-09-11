"""General-capability and task-performance eval.

Spec section 6: "General capability — MMLU subset, to separate targeted
safety loss from broad forgetting" and "Task performance — held-out task
metric." Written against the same Generator protocol agent.Agent.generate
already implements, so this plugs into any agent without either side
needing to change.
"""
from __future__ import annotations

import re
import string
from typing import Any, Protocol


class Generator(Protocol):
    """Whatever agent.py exposes for batched generation, e.g. a bound
    method on Agent wrapping HF `generate` or vLLM."""

    def __call__(self, prompts: list[str]) -> list[str]: ...


_MMLU_LETTERS = ["A", "B", "C", "D"]


def _format_mmlu_prompt(question: str, choices: list[str]) -> str:
    lettered = "\n".join(f"{letter}. {choice}" for letter, choice in zip(_MMLU_LETTERS, choices))
    return f"{question}\n{lettered}\nAnswer with just the letter."


def _parse_mmlu_answer(generation: str) -> str | None:
    """First bare A/B/C/D in the response, word-bounded so it doesn't
    match a letter inside an ordinary word. None if the model didn't
    give a parseable answer — excluded from accuracy, not scored wrong."""
    match = re.search(r"\b([ABCD])\b", generation.strip().upper())
    return match.group(1) if match else None


def run_mmlu_subset(generate: Generator, subset_size: int, seed: int = 0) -> dict[str, Any]:
    """Load an MMLU subset (HF `cais/mmlu`, "all" config — ungated), format
    each question as a lettered multiple-choice prompt, generate, and
    score exact-match on the answer letter.

    Returns {"n_total", "n_unparseable", "n_scored", "n_correct", "accuracy"}.
    """
    import random

    import datasets

    ds = datasets.load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(subset_size, len(ds)))
    examples = [ds[i] for i in indices]

    prompts = [_format_mmlu_prompt(ex["question"], ex["choices"]) for ex in examples]
    generations = generate(prompts)

    n_unparseable = 0
    n_correct = 0
    for ex, gen in zip(examples, generations):
        predicted = _parse_mmlu_answer(gen)
        if predicted is None:
            n_unparseable += 1
            continue
        correct_letter = _MMLU_LETTERS[ex["answer"]]
        if predicted == correct_letter:
            n_correct += 1

    n_scored = len(examples) - n_unparseable
    return {
        "n_total": len(examples),
        "n_unparseable": n_unparseable,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_scored) if n_scored else 0.0,
    }


def _token_f1(prediction: str, reference: str) -> float:
    """Whitespace-tokenized precision/recall F1 between a generation and
    its reference — the SQuAD-style overlap metric. Dependency-free (no
    rouge-score/nltk), which is enough to catch "did benign task
    performance collapse," the thing this metric is actually for; it
    isn't trying to be a publication-grade generation metric."""
    strip_punct = str.maketrans("", "", string.punctuation)
    pred_tokens = prediction.lower().translate(strip_punct).split()
    ref_tokens = reference.lower().translate(strip_punct).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common: dict[str, int] = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in common.items():
        overlap += min(c, ref_counts.get(t, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def run_held_out_task(generate: Generator, held_out_examples: list[dict]) -> dict[str, Any]:
    """Score the task's held-out split (Alpaca-format {instruction, input,
    output} examples never used for training) via token-overlap F1
    against the reference output — separates "did the agent forget the
    benign task" from "did safety erode," per spec section 6.

    Returns {"n_examples", "mean_f1"}.
    """
    prompts = []
    for ex in held_out_examples:
        instruction = ex["instruction"]
        if ex.get("input"):
            instruction = f"{instruction}\n\n{ex['input']}"
        prompts.append(instruction)

    generations = generate(prompts)
    f1_scores = [_token_f1(gen, ex["output"]) for gen, ex in zip(generations, held_out_examples)]
    return {
        "n_examples": len(held_out_examples),
        "mean_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }
