# p2p-safety

Safety erosion in peer-to-peer federated learning: does benign fine-tuning
degrade refusal behavior, does that damage spread when agents average LoRA
adapters over a P2P network, and can a few safety-holding peers repair it?
See [`../P2P_Safety_Project_SPEC.md`](../P2P_Safety_Project_SPEC.md) for
the full research spec — motivation, research questions, protocol,
metrics, and the phased build plan this repo follows.

## Status

Phase 0 scaffold. Repo structure, Hydra config tree, and all `src/`
modules are in place. Everything that doesn't require a GPU or the
model-loading stack (graph construction, both averaging modes, data
partitioning, the substring refusal scorer, the round-loop orchestration
logic) is implemented and tested. `agent.py`'s model-loading, training,
and generation methods, and the eval/analysis code that depends on real
model output, are Phase 1 stubs — see their docstrings.

## Setup

Two tiers of dependencies (`requirements.txt`):

- Lightweight (numpy, networkx, pyyaml, pytest) — enough to run the test
  suite and everything that doesn't touch a model.
- Full stack (torch, transformers, peft, trl, datasets, hydra-core,
  wandb) — needed from Phase 1 onward.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # full stack; skip if only running tests
```

## Tests

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

Covers (spec section 9's required list): neighbor-set correctness,
SVD refactor error bound, identical-data / repeated-averaging convergence,
the 2-agent-complete-graph-reduces-to-plain-averaging check, train/eval
prompt disjointness, and seed determinism.

## Repo structure

See spec section 7. Model weights, checkpoints, datasets, and `results/`
are gitignored — run history lives in Weights & Biases, traced back to a
git commit SHA (`p2p_safety.utils.get_git_sha`) logged with every run.

## Running an experiment

Once the full stack is installed (Phase 1+):

```bash
python scripts/run_experiment.py experiment=single_agent
python scripts/run_experiment.py experiment=p2p_network graph=random
python scripts/sweep.py --multirun graph=ring,random,complete seed=0,1,2
```
