"""Agent: LoRA state and the local train step.

Needs the real model-loading stack (torch/transformers/peft/trl), which
this environment doesn't have installed (see conf/ and requirements.txt
for the intended stack). The class below defines the interface simulate.py
and average.py are written against; fill in the bodies in Phase 1 without
changing that interface.

Do not import p2p_safety.data.safety_data here — see that module's
docstring. Local training uses task_data only.
"""
from __future__ import annotations

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

    # Populated once the model/adapter are loaded (Phase 1).
    _model: Any = None
    _tokenizer: Any = None

    def load(self, base_model_name: str, device: str, dtype: str, seed: int) -> None:
        """Load the shared base model (once, ideally shared across agents
        to save memory) and attach a freshly initialized LoRA adapter.

        Phase 1. Should end with self._model being a PeftModel with an
        adapter initialized from `seed`, so re-running with the same seed
        reproduces the same starting adapter (spec section 9, determinism).
        """
        raise NotImplementedError("Phase 1: transformers + peft model loading")

    def get_adapter_state(self) -> Adapter:
        """Extract the current LoRA A/B matrices as the Adapter dict shape
        average.py operates on (module_name -> {"A": arr, "B": arr}).
        Phase 1: read from self._model's PeftModel state_dict.
        """
        raise NotImplementedError("Phase 1: extract A/B from PeftModel state_dict")

    def set_adapter_state(self, adapter: Adapter) -> None:
        """Load an Adapter dict (e.g. the result of average.average_adapters)
        back into self._model's LoRA layers."""
        raise NotImplementedError("Phase 1: write A/B into PeftModel state_dict")

    def local_train_step(self, n_steps: int, lr: float, batch_size: int, seed: int) -> dict[str, float]:
        """Run E steps of LoRA SFT on self.train_data (+ safety_examples if
        is_safety_holding). Returns a dict of scalar metrics (e.g. mean
        loss) for logging.

        Phase 1: trl SFTTrainer or a plain torch loop over self.train_data.
        Must not touch anything from p2p_safety.data.safety_data.
        """
        raise NotImplementedError("Phase 1: LoRA SFT training loop")

    def generate(self, prompts: list[str], max_new_tokens: int, do_sample: bool) -> list[str]:
        """Batched generation for eval (safety.py / capability.py call
        this). Phase 1: batched HF `generate`, or vLLM if it fits (spec
        section 5)."""
        raise NotImplementedError("Phase 1: batched generation")
