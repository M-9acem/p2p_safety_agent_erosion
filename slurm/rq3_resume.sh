#!/bin/bash
#SBATCH --job-name=p2p_rq3_resume
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
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

# The first sbatch (rq3_sweep.sh, job 1051010) capped --time at 14h,
# which turned out to be nowhere near enough — 16-agent + full_metrics
# rounds run at ~45min/round, so 40 rounds needs ~30h, not 14h. Wave 1
# (tasks 0-3, k=1) got to round 17-18/40 before SLURM cancelled it at
# the 14h mark; wave 2 (tasks 4-7, k=2) is running the same course;
# wave 3 (tasks 8-11, k=4) never started (cancelled here before wasting
# any GPU time on a run we already know would hit the same wall).
#
# This resubmission resumes wave 1 + wave 2 from their checkpoints
# (same hydra.run.dir as the original run, so new rounds append to the
# same round*_asr.json sequence rather than starting a fresh directory)
# and starts wave 3 fresh, all with a time budget that actually covers
# a full 40-round run (36h, vs. the ~31h worst case).
#
# --dependency=afterany:1051010 (set at submission time, not here) is
# what keeps this at <=4 concurrent jobs overall: it won't even be
# considered by the scheduler until every task of the original sweep
# (1051010) has reached a terminal state, so it can never overlap with
# what's still running there.

K_VALUES=(1 1 1 1 2 2 2 2 4 4 4 4)
PLACEMENTS=(well_connected peripheral clustered spread well_connected peripheral clustered spread well_connected peripheral clustered spread)

k="${K_VALUES[$SLURM_ARRAY_TASK_ID]}"
placement="${PLACEMENTS[$SLURM_ARRAY_TASK_ID]}"

if [ "$SLURM_ARRAY_TASK_ID" -lt 8 ]; then
  # resume wave 1 (0-3) or wave 2 (4-7) in place
  run_dir="$SLURM_SUBMIT_DIR/outputs/rq3/task${SLURM_ARRAY_TASK_ID}_k${k}_${placement}_job1051010"
  resume_from="$run_dir/p2p_network/checkpoints/latest"
  echo "rq3 resume task $SLURM_ARRAY_TASK_ID: k=$k placement=$placement resuming from $resume_from"
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
    experiment.resume_from="$resume_from" \
    hydra.run.dir="$run_dir"
else
  # wave 3 (8-11): fresh, was cancelled before it ever started
  run_dir="$SLURM_SUBMIT_DIR/outputs/rq3/task${SLURM_ARRAY_TASK_ID}_k${k}_${placement}"
  echo "rq3 resume task $SLURM_ARRAY_TASK_ID: k=$k placement=$placement fresh start"
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
fi
