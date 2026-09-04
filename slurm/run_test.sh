#!/bin/bash
#SBATCH --job-name=p2p_safety_single_agent
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --constraint='GPURAM_Min_16GB'
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source /etc/profile.d/conda.sh
conda activate "$HOME/conda_envs/p2p_safety"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$SLURM_SUBMIT_DIR/.cache/huggingface"
# stdout is block-buffered when it's a file, not a tty — without this,
# print()-based progress (loss/ASR per step) only shows up once the whole
# job exits, which defeats checking on a long run mid-flight.
export PYTHONUNBUFFERED=1

# What to run — one line, easy to swap as phases progress.
# Phase 0 (done, tagged phase-0-baseline-established): baseline safety
# eval on the untouched model.
# python scripts/run_experiment.py experiment=baseline

# Phase 1 (current): single-agent LoRA SFT erosion.
python scripts/run_experiment.py experiment=single_agent

# Phase 3+ (once single-agent erosion reproduces):
# python scripts/run_experiment.py experiment=p2p_network graph=random
