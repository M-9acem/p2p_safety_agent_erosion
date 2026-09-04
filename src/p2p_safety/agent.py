"""Agent: LoRA state and the local train step.

Do not import p2p_safety.data.safety_data here — see that module's
docstring. Local training uses task_data only.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from p2p_safety.average import Adapter


@dataclass
class Agent:
    agent_id: int
    model_name: str
    lora_config: dict[str, Any]
    train_data: list[dict[str, Any]]
    is_safety_holding: bool = False
    safety_examples: list[dict[str, Any]] = field(default_factory=list)

    # Populated by load().
    _model: Any = None
    _tokenizer: Any = None
    _optimizer: Any = None
    _device: Any = None

    def load(self, base_model_name: str, device: str, dtype: str, seed: int) -> None:
        """Load the base model + tokenizer and attach a freshly initialized
        LoRA adapter. Seeding torch before get_peft_model makes the LoRA A
        matrix's random init (B starts at zero, per PEFT convention)
        reproducible for a given seed (spec section 9)."""
        import torch
        from peft import LoraConfig, get_peft_model

        from p2p_safety.model_utils import load_model_and_tokenizer
        from p2p_safety.utils import seed_everything

        seed_everything(seed)
        model, tokenizer = load_model_and_tokenizer(base_model_name, dtype, device)

        peft_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config["dropout"],
            target_modules=list(self.lora_config["target_modules"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.train()

        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def _lora_layers(self) -> dict[str, Any]:
        """PEFT-wrapped Linear layers targeted by LoRA, keyed by their full
        dotted module path — the granularity average.py's Adapter dict
        needs (one A/B pair per targeted layer instance, not per module
        *type*: a 28-layer model has 28 separate q_proj adapters)."""
        return {
            name: module
            for name, module in self._model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        }

    def get_adapter_state(self) -> Adapter:
        adapter: Adapter = {}
        for name, layer in self._lora_layers().items():
            adapter[name] = {
                "A": layer.lora_A["default"].weight.detach().cpu().float().numpy(),
                "B": layer.lora_B["default"].weight.detach().cpu().float().numpy(),
            }
        return adapter

    def set_adapter_state(self, adapter: Adapter) -> None:
        import torch

        layers = self._lora_layers()
        for name, layer in layers.items():
            if name not in adapter:
                raise ValueError(f"adapter missing layer {name!r} present in this agent's model")
            a_weight = layer.lora_A["default"].weight
            b_weight = layer.lora_B["default"].weight
            with torch.no_grad():
                a_weight.copy_(torch.from_numpy(adapter[name]["A"]).to(a_weight.device, a_weight.dtype))
                b_weight.copy_(torch.from_numpy(adapter[name]["B"]).to(b_weight.device, b_weight.dtype))

    def _format_example(self, example: dict[str, Any]) -> tuple[str, str]:
        """Alpaca-style {instruction, input, output} -> (prompt, response)."""
        instruction = example["instruction"]
        if example.get("input"):
            instruction = f"{instruction}\n\n{example['input']}"
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, example["output"]

    def local_train_step(self, n_steps: int, lr: float, batch_size: int, seed: int) -> dict[str, float]:
        """E steps of LoRA SFT: response-only loss (prompt tokens masked
        to -100) on self.train_data, plus self.safety_examples for a
        safety-holding agent (spec section 5, RQ3 data). Never touches
        p2p_safety.data.safety_data."""
        import torch

        model, tokenizer = self._model, self._tokenizer
        device = self._device
        model.train()

        if self._optimizer is None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.AdamW(trainable, lr=lr)
        # lr can change between calls (e.g. a config sweep); keep it in sync.
        for group in self._optimizer.param_groups:
            group["lr"] = lr

        pool = list(self.train_data) + (list(self.safety_examples) if self.is_safety_holding else [])
        rng = random.Random(seed * 1_000_003 + self.agent_id)

        losses = []
        for _ in range(n_steps):
            batch = [rng.choice(pool) for _ in range(batch_size)]
            input_ids_list, labels_list = [], []
            for ex in batch:
                prompt, response = self._format_example(ex)
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
                ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + response_ids
                input_ids_list.append(ids)
                labels_list.append(labels)

            max_len = max(len(ids) for ids in input_ids_list)
            pad_id = tokenizer.pad_token_id
            input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
            labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
            attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
            for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
                # right-pad: training doesn't need generation's left-padding
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
                attention_mask[i, : len(ids)] = 1

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        return {"mean_loss": sum(losses) / len(losses), "n_steps": n_steps}

    def generate(self, prompts: list[str], max_new_tokens: int, do_sample: bool) -> list[str]:
        from p2p_safety.model_utils import batched_generate

        self._model.eval()
        return batched_generate(
            self._model,
            self._tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            batch_size=16,
            device=self._device,
        )
