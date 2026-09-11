# p2p-safety

Safety erosion in peer-to-peer federated learning: does benign fine-tuning
degrade refusal behavior, does that damage spread when agents average LoRA
adapters over a P2P network, and can a few safety-holding peers repair it?
See [`../P2P_Safety_Project_SPEC.md`](../P2P_Safety_Project_SPEC.md) for
the full research spec — motivation, research questions, protocol,
metrics, and the phased build plan this repo follows.

## Status

**Phases 0–3 complete**, all on real trained models (Qwen2.5-1.5B-Instruct,
LoRA r=8) on a SLURM/GPU cluster — not just the unit tests. Tagged
milestones: `phase-0-baseline-established`, `phase-1-erosion-reproduced`,
`phase-1-classifier-scorer-fixed`, `phase-2-p2p-machinery-validated`.

**Headline result** (Phase 3, 18-run sweep — {ring, random, complete} ×
{8, 16} agents × 3 seeds, 40 rounds each): safety erosion happens and
spreads through *every* configuration, from a 0.4% baseline to 50–74%
final ASR (RQ1 — clear yes). Connectivity density does **not** show a
statistically distinguishable effect on erosion magnitude — the spread
between topologies (3.5pp) is smaller than the seed-to-seed spread within
a single topology (up to 6.7pp) (RQ2 — no clear effect, reported honestly
rather than dressed up).

**Full analysis with plots and tables:** [`notebooks/results_analysis.ipynb`](notebooks/results_analysis.ipynb).

RQ3 (repair via safety-holding peers) is the natural next phase — the
config path (`data.safety_holding`) is already wired into
`_run_p2p_network`, just never exercised at scale.

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

```bash
python scripts/run_experiment.py experiment=baseline
python scripts/run_experiment.py experiment=single_agent
python scripts/run_experiment.py experiment=p2p_network graph=random graph.n_agents=8 seed=0
```

Cluster-specific tooling (SLURM job script, queue-managed sweep runner,
results-pull script) lives in [`slurm/`](slurm/).

## Results notebook

```bash
pip install -r requirements.txt   # or just matplotlib/pandas/jupyter/nbconvert/ipykernel
jupyter nbconvert --to notebook --execute --inplace notebooks/results_analysis.ipynb
```

Reads `outputs/` (Phases 0–2) and `slurm/logs/*.out` (Phase 3 — the
source of truth for the sweep; see the notebook's own note on why, not
the timestamp-collision-prone `outputs/` dirs) directly — no hand-entered
numbers anywhere in it.
