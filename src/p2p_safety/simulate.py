"""The round loop (spec section 4): the whole P2P algorithm.

Each round t, for every agent:
  1. LOCAL TRAIN — E steps of LoRA SFT on its own data shard
  2. EXCHANGE    — send its adapter to its neighbors
  3. AVERAGE     — replace its adapter with the uniform average of its own
                   and all received neighbor adapters

Every K rounds: evaluate every agent on safety, task, and general
capability (left to scripts/run_experiment.py, which owns W&B logging and
checkpointing; this module is pure simulation logic so it's testable
without either).

Written against a minimal protocol (get_adapter_state / set_adapter_state /
local_train_step) rather than the concrete Agent class, so Phase 2's
acceptance checks (identical-data convergence, 2-agent complete graph
reduces to plain averaging) can run against lightweight test doubles
without the model-loading stack.
"""
from __future__ import annotations

from typing import Any, Protocol

import networkx as nx

from p2p_safety.average import Adapter, average_adapters
from p2p_safety.graph import closed_neighborhood


class RoundAgent(Protocol):
    agent_id: int

    def get_adapter_state(self) -> Adapter: ...
    def set_adapter_state(self, adapter: Adapter) -> None: ...
    def local_train_step(self, n_steps: int, lr: float, batch_size: int, seed: int) -> dict[str, float]: ...


def run_round(
    g: nx.Graph,
    agents: dict[int, RoundAgent],
    round_idx: int,
    local_steps: int,
    lr: float,
    batch_size: int,
    average_mode: str,
    lora_alpha: float | None = None,
    lora_r: int | None = None,
    seed: int = 0,
) -> dict[int, dict[str, Any]]:
    """Run one round: LOCAL TRAIN, EXCHANGE, AVERAGE for every agent.

    EXCHANGE and AVERAGE are combined here: since averaging is a pure
    function of the closed neighborhood's adapter states, "sending" is just
    reading get_adapter_state() from each neighbor — there's no separate
    wire step to simulate on one process.

    Returns per-agent info: {"train_metrics": ..., "average_info": ...}.
    Every agent's post-train state is snapshotted before ANY agent is
    updated with its averaged state, so round t+1's neighbors all average
    over round t's post-train (not partially-averaged) adapters —
    synchronous rounds, per spec section 4.
    """
    round_info: dict[int, dict[str, Any]] = {}

    # 1. LOCAL TRAIN
    post_train_state: dict[int, Adapter] = {}
    for agent_id, agent in agents.items():
        metrics = agent.local_train_step(
            n_steps=local_steps, lr=lr, batch_size=batch_size, seed=seed + round_idx
        )
        post_train_state[agent_id] = agent.get_adapter_state()
        round_info[agent_id] = {"train_metrics": metrics}

    # 2. EXCHANGE + 3. AVERAGE, all against the pre-averaging snapshot above
    for agent_id, agent in agents.items():
        neighborhood = closed_neighborhood(g, agent_id)
        neighbor_states = [post_train_state[n] for n in neighborhood]
        new_state, info = average_adapters(
            neighbor_states, mode=average_mode, alpha=lora_alpha, r=lora_r
        )
        agent.set_adapter_state(new_state)
        round_info[agent_id]["average_info"] = info

    return round_info


def run_simulation(
    g: nx.Graph,
    agents: dict[int, RoundAgent],
    n_rounds: int,
    local_steps: int,
    lr: float,
    batch_size: int,
    average_mode: str,
    eval_every_round: int,
    eval_fn,
    lora_alpha: float | None = None,
    lora_r: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Drive n_rounds of run_round, calling eval_fn(round_idx, agents)
    every eval_every_round rounds (and after the final round). Returns the
    list of eval_fn results collected."""
    eval_log = []
    for round_idx in range(n_rounds):
        run_round(
            g,
            agents,
            round_idx,
            local_steps,
            lr,
            batch_size,
            average_mode,
            lora_alpha,
            lora_r,
            seed,
        )
        is_last = round_idx == n_rounds - 1
        if eval_every_round and (round_idx % eval_every_round == 0 or is_last):
            eval_log.append(eval_fn(round_idx, agents))
    return eval_log
