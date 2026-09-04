"""Agent: LoRA state and the local train step.

Do not import p2p_safety.data.safety_data here — see that module's
docstring. Local training uses task_data only.

Phase 2: N agents share ONE base model rather than each loading their own
full copy — loading N separate 1.5B-parameter models for an N-agent P2P
run doesn't fit, and isn't needed, since only the (tiny) LoRA adapters
differ per agent. PEFT supports multiple named adapters on one base model
directly (add_adapter / set_adapter); build_shared_multi_agent_model sets
that up once, and each Agent just holds its own adapter_name into it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from p2p_safety.average import Adapter


def build_shared_multi_agent_model(
    model: Any, lora_config: dict[str, Any], agent_ids: list[int], seeds: dict[int, int]
) -> Any:
    """Wrap a freshly loaded (not yet PEFT-wrapped) causal LM with one
    independently seeded LoRA adapter per agent id, all on the same
    frozen base weights. Returns the shared PeftModel; pass it (plus the
    tokenizer) to each Agent.bind().

    Seeding per agent before its adapter is added reproduces that agent's
    specific random LoRA-A init for a given seed (spec section 9),
    independent of how many other agents were added before it.
    """
    from peft import LoraConfig, get_peft_model

    from p2p_safety.utils import seed_everything

    def make_peft_config() -> Any:
        return LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=list(lora_config["target_modules"]),
            task_type="CAUSAL_LM",
        )

    if not agent_ids:
        raise ValueError("agent_ids must be non-empty")

    first_id, *rest_ids = agent_ids
    seed_everything(seeds[first_id])
    peft_model = get_peft_model(model, make_peft_config(), adapter_name=f"agent_{first_id}")
    for aid in rest_ids:
        seed_everything(seeds[aid])
        peft_model.add_adapter(f"agent_{aid}", make_peft_config())
    peft_model.train()
    return peft_model


@dataclass
class Agent:
    agent_id: int
    lora_config: dict[str, Any]
    train_data: list[dict[str, Any]]
    is_safety_holding: bool = False
    safety_examples: list[dict[str, Any]] = field(default_factory=list)

    # Populated by load() (standalone, Phase 1) or bind() (shared, Phase 2+).
    _model: Any = None
    _tokenizer: Any = None
    _optimizer: Any = None
    _device: Any = None
    _adapter_name: str = "default"

    def load(self, base_model_name: str, device: str, dtype: str, seed: int) -> None:
        """Phase 1 path: this agent gets its own full model + adapter, no
        sharing. Fine for one agent; don't use this in a multi-agent run —
        use build_shared_multi_agent_model + bind() instead."""
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
        self._adapter_name = "default"

    def bind(self, shared_model: Any, tokenizer: Any, device: str) -> None:
        """Phase 2+ path: attach to a model built by
        build_shared_multi_agent_model, using this agent's own adapter_name
        (already added to shared_model) for everything below."""
        self._model = shared_model
        self._tokenizer = tokenizer
        self._device = device
        self._adapter_name = f"agent_{self.agent_id}"

    def _activate(self) -> None:
        """Switch the shared model's active adapter to this agent's,
        before any forward pass (train or generate). A no-op cost-wise —
        just flips which adapter's LoRA delta gets added in forward."""
        self._model.set_adapter(self._adapter_name)

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

    def trainable_parameters(self) -> list[Any]:
        """This agent's own adapter parameters only — with multiple named
        adapters sharing one model, every adapter's params default to
        requires_grad=True, so an optimizer built from
        model.parameters() would silently update every agent's adapter
        at once instead of just this one's."""
        params = []
        for layer in self._lora_layers().values():
            params.extend(layer.lora_A[self._adapter_name].parameters())
            params.extend(layer.lora_B[self._adapter_name].parameters())
        return params

    def get_adapter_state(self) -> Adapter:
        adapter: Adapter = {}
        for name, layer in self._lora_layers().items():
            adapter[name] = {
                "A": layer.lora_A[self._adapter_name].weight.detach().cpu().float().numpy(),
                "B": layer.lora_B[self._adapter_name].weight.detach().cpu().float().numpy(),
            }
        return adapter

    def set_adapter_state(self, adapter: Adapter) -> None:
        import torch

        layers = self._lora_layers()
        for name, layer in layers.items():
            if name not in adapter:
                raise ValueError(f"adapter missing layer {name!r} present in this agent's model")
            a_weight = layer.lora_A[self._adapter_name].weight
            b_weight = layer.lora_B[self._adapter_name].weight
            with torch.no_grad():
                a_weight.copy_(torch.from_numpy(adapter[name]["A"]).to(a_weight.device, a_weight.dtype))
                b_weight.copy_(torch.from_numpy(adapter[name]["B"]).to(b_weight.device, b_weight.dtype))

    def save_checkpoint(self, path: str) -> None:
        """Adapter state only (spec section 9: checkpoint every round so a
        run resumes rather than restarts). Optimizer momentum isn't
        preserved — a resumed run's early steps behave slightly
        differently from an uninterrupted one, a known, acceptable gap."""
        import numpy as np

        adapter = self.get_adapter_state()
        flat = {}
        for name, ab in adapter.items():
            flat[f"{name}::A"] = ab["A"]
            flat[f"{name}::B"] = ab["B"]
        np.savez(path, **flat)

    def load_checkpoint(self, path: str) -> None:
        import numpy as np

        loaded = np.load(path)
        adapter: Adapter = {}
        for key in loaded.files:
            name, which = key.rsplit("::", 1)
            adapter.setdefault(name, {})[which] = loaded[key]
        self.set_adapter_state(adapter)

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

        self._activate()
        model, tokenizer = self._model, self._tokenizer
        device = self._device
        model.train()

        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=lr)
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

        self._activate()
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
