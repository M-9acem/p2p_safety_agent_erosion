"""Main entry point: one run, fully specified by its Hydra config + seed
(spec section 9). Requires the full stack (torch/transformers/peft/trl/
hydra/wandb) which is not installed in this scaffold — see
requirements.txt. Wire up phase by phase rather than writing it all before
Phase 1 reproduces the single-agent result (spec section 11).

Usage (once deps are installed):
    python scripts/run_experiment.py experiment=single_agent
    python scripts/run_experiment.py experiment=p2p_network graph=random
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from p2p_safety.utils import get_git_sha, seed_everything


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    import wandb

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.config.update({"git_sha": get_git_sha()})

    if cfg.experiment.name == "single_agent":
        _run_single_agent(cfg)
    elif cfg.experiment.name == "p2p_network":
        _run_p2p_network(cfg)
    else:
        raise ValueError(f"unknown experiment: {cfg.experiment.name}")


def _run_single_agent(cfg: DictConfig) -> None:
    # Phase 1: load one Agent, run local_train_step in a loop, run safety
    # eval every cfg.experiment.eval_every_steps, log to wandb.
    raise NotImplementedError("Phase 1")


def _run_p2p_network(cfg: DictConfig) -> None:
    # Phase 3+: build the graph, load N agents (sharing the base model),
    # partition data across them (+ safety-holding placement for RQ3),
    # drive simulate.run_simulation with an eval_fn that logs to wandb and
    # checkpoints every round (spec section 9: "checkpoint every round").
    raise NotImplementedError("Phase 2/3")


if __name__ == "__main__":
    main()
