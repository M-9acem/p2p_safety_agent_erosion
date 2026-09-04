#!/bin/bash
# Pull results/data back from the cluster after a run — JSON, logs, and
# Hydra configs only, never model weights or the HF/dataset cache (those
# stay on the cluster; --dry-run first if unsure what would transfer).
#
# Usage: slurm/pull_results.sh [--dry-run]

set -euo pipefail

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN="-n"
fi

cd "$(dirname "$0")/.."

rsync -avz $DRY_RUN \
  --include='*/' --include='*.json' --include='*.txt' --include='*.yaml' --include='*.log' \
  --exclude='*.pt' --exclude='*.bin' --exclude='*.safetensors' --exclude='*' \
  skacem@Hades.univ-avignon.fr:/users/skacem/p2p-safety/outputs/ outputs/
