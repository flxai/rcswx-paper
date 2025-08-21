#!/usr/bin/env bash
# Please use scripts/slurm-split.sh beforehand to split jobs into single files.
set -euo pipefail
shopt -s nullglob

BASE_DIR="${BASE_DIR:-$HOME/einsearch}"
JOB_DIR="$BASE_DIR/jobs"
SCRIPT_PATH="$BASE_DIR/scripts/slurm-run.sh"

# Optional arg: glob pattern of job files; default matches split1 jobs.
PATTERN="${1:-$JOB_DIR/evolution_config_*_split1_*}"

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
