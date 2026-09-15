#!/bin/bash
#SBATCH --job-name=p2p_rq3_resume2
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source /etc/profile.d/conda.sh
conda activate "$HOME/conda_envs/p2p_safety"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$SLURM_SUBMIT_DIR/.cache/huggingface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# rq3_resume.sh's task 8 (k=4, well_connected) hit its 36h limit at
# round 36/40 -- 4 rounds short, the array's other 11 tasks all
# finished. Only this one needs a second resume.
run_dir="$SLURM_SUBMIT_DIR/outputs/rq3/task8_k4_well_connected"
resume_from="$run_dir/p2p_network/checkpoints/latest"

python scripts/run_experiment.py \
  experiment=p2p_network \
  graph=random graph.n_agents=16 \
  seed=0 \
  data.safety_holding.enabled=true \
  data.safety_holding.n_agents=4 \
  data.safety_holding.placement=well_connected \
  experiment.full_metrics=true \
  experiment.n_rounds=40 \
  experiment.local_steps=20 \
  experiment.resume_from="$resume_from" \
  hydra.run.dir="$run_dir"
