#!/bin/bash
#SBATCH --job-name=p2p_safety_single_agent
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
# GPURAM_Min_16GB looked sufficient in early smoke tests but isn't: a
# "16GB-class" node reports ~15.77 GiB usable, and 9 of Phase 3's first
# 18 sweep jobs OOM'd on exactly that tier while the same config ran fine
# for 10+ hours on 24GB+ cards (RTX 3090 / A100 / V100-32GB). Require the
# margin that's actually validated.
#SBATCH --constraint='GPURAM_Min_24GB'
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
# PyTorch's own suggestion on the Phase 3 OOMs: several failed with a lot
# of "reserved but unallocated" memory (fragmentation), not just being
# genuinely out of room.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# What to run — one line, easy to swap as phases progress.
# Phase 0 (done, tagged phase-0-baseline-established): baseline safety
# eval on the untouched model.
# python scripts/run_experiment.py experiment=baseline

# Phase 1 (current): single-agent LoRA SFT erosion.
python scripts/run_experiment.py experiment=single_agent

# Phase 3+ (once single-agent erosion reproduces):
# python scripts/run_experiment.py experiment=p2p_network graph=random
