"""Model loading and batched generation, shared by the Phase 0 baseline
eval and Agent (Phase 1+).

Requires torch + transformers; not importable in the numpy-only scaffold
environment, only on a machine with the full stack installed (see
requirements.txt / README.md).
"""
from __future__ import annotations

from typing import Any

_DTYPE_MAP = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_model_and_tokenizer(model_name: str, dtype: str, device: str) -> tuple[Any, Any]:
    """Load a causal LM + tokenizer for eval/generation.

    Args:
        model_name: HF hub id, e.g. "Qwen/Qwen2.5-1.5B-Instruct".
        dtype: one of "bf16" / "fp16" / "fp32" (spec section 5 config).
        device: "cuda" or "cpu".
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, _DTYPE_MAP[dtype])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for batched generation so every sequence's real content ends
    # at the same position and `generate` doesn't need per-row trimming.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def batched_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    batch_size: int,
    device: str,
) -> list[str]:
    """Chat-format each prompt as a single user turn, generate, and return
    only the newly generated text (prompt stripped).

    Instruct models need the chat template applied — feeding a raw prompt
    without it produces off-distribution completions that don't reflect
    the model's actual (aligned) behavior, which would silently corrupt
    every downstream ASR number.
    """
    import torch

    outputs: list[str] = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
    return outputs
