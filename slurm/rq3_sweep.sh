#!/bin/bash
#SBATCH --job-name=p2p_rq3
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --array=0-11%4
#SBATCH --output=slurm/logs/%x_%a_%j.out
#SBATCH --error=slurm/logs/%x_%a_%j.err

set -euo pipefail

source /etc/profile.d/conda.sh
conda activate "$HOME/conda_envs/p2p_safety"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$SLURM_SUBMIT_DIR/.cache/huggingface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RQ3 / Phase 4 (spec section 8): k in {1,2,4} safety-holding agents x
# 4 placement strategies, on the same random/16-agent/seed=0 graph as
# phase 3's random_16_s0 run (asr_mean=0.6737 at round 39) — that run
# is this sweep's k=0 baseline, no need to rerun it.
#
# ring and complete topologies are degree-regular (every node has the
# same degree), which makes "well_connected" vs "peripheral" placement
# degenerate to an arbitrary tie-break — only "random" has the degree
# variance the placement comparison actually needs, and it also gives
# a non-trivial hop-distance spread for the second headline plot.
#
# #SBATCH --array=0-11%4 caps this at <=4 concurrent jobs by itself
# (SLURM throttles the array, no manual batching needed).

K_VALUES=(1 1 1 1 2 2 2 2 4 4 4 4)
PLACEMENTS=(well_connected peripheral clustered spread well_connected peripheral clustered spread well_connected peripheral clustered spread)

k="${K_VALUES[$SLURM_ARRAY_TASK_ID]}"
placement="${PLACEMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "rq3 sweep task $SLURM_ARRAY_TASK_ID: k=$k placement=$placement"

# Explicit run dir, not hydra's default now-timestamp one: phase 3's
# sweep had two jobs land in the same outputs/ dir when they started in
# the same second, and only the SLURM job name saved it after the fact.
# Naming this by array task id upfront means it can't happen here.
run_dir="$SLURM_SUBMIT_DIR/outputs/rq3/task${SLURM_ARRAY_TASK_ID}_k${k}_${placement}_job${SLURM_ARRAY_JOB_ID}"

python scripts/run_experiment.py \
  experiment=p2p_network \
  graph=random graph.n_agents=16 \
  seed=0 \
  data.safety_holding.enabled=true \
  data.safety_holding.n_agents="$k" \
  data.safety_holding.placement="$placement" \
  experiment.full_metrics=true \
  experiment.n_rounds=40 \
  experiment.local_steps=20 \
  hydra.run.dir="$run_dir"
