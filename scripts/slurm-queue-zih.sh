#!/usr/bin/env bash
# Please use scripts/slurm-split.sh before accordingly to split jobs into single files
set -euo pipefail
shopt -s nullglob

BASE_DIR="$HOME/einsearch"
JOB_DIR="$BASE_DIR/jobs"
SCRIPT_PATH="$BASE_DIR/scripts/slurm-run.sh"

echo "Queueing jobs..."
for fn in "$JOB_DIR/"evolution_config_*_split1_*; do
  echo "$fn"
  sbatch \
    -vvv \
    -D "$BASE_DIR" \
    --export=ALL \
    --time=48:00:00 \
    --cpus-per-task=6 \
    --gres=gpu:1 \
    --mem=120G \
    --ntasks=1 \
    --nodes=1 \
      "$SCRIPT_PATH" \
      "$fn" \
      1
done
