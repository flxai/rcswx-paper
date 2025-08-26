#!/usr/bin/env bash
# Please use scripts/slurm-split.sh beforehand to split jobs into single files.
set -euo pipefail
shopt -s nullglob

BASE_DIR="${BASE_DIR:-$HOME/einsearch}"
JOB_DIR="$BASE_DIR/jobs"
SCRIPT_PATH="$BASE_DIR/scripts/slurm-run.sh"
# Slurm account/budget (override via env)
BUDGET="${BUDGET:-hai_1097}"

# Optional arg: glob pattern; default matches seed_1_generational jobs.
PATTERN="${1:-$JOB_DIR/evolution_config_seed_1_generational_*}"

# Expand pattern robustly (handles no-match without failing).
mapfile -t JOB_FILES < <(compgen -G "$PATTERN" || true)
if ((${#JOB_FILES[@]} == 0)); then
  echo "No jobs matched pattern: $PATTERN" >&2
  exit 1
fi

echo "Queueing ${#JOB_FILES[@]} jobs matching: $PATTERN"
for fn in "${JOB_FILES[@]}"; do
  echo "$fn"
  sbatch \
    -D "$BASE_DIR" \
    -A "$BUDGET" \
    --export=ALL \
    --time=24:00:00 \
    --ntasks=1 \
    --gpus-per-task=4 \
    "$SCRIPT_PATH" \
    "$fn" \
    4
done
