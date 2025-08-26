#!/usr/bin/env bash
# Queue JUWELS Booster jobs (GPU). After creating per-job config files.
set -euo pipefail
shopt -s nullglob

# Defaults for JUWELS
PROJECT="${PROJECT:-hai_1006}"               # override if needed
PARTITION="${PARTITION:-booster}"            # Booster (A100)
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

BASE_DIR="${BASE_DIR:-$HOME/einsearch}"
JOB_DIR="$BASE_DIR/jobs"
SCRIPT_PATH="$BASE_DIR/scripts/slurm-run.sh"

# Budget: prefer env; else derive from `jutil user projects`
if [[ -z "${BUDGET:-}" ]]; then
  BUDGET="$(jutil user projects --noheader | awk -v p="$PROJECT" '$1==p{print $NF; exit}')"
fi
: "${BUDGET:?No JUWELS budget for project '$PROJECT'. Run 'jutil user projects' and use the value in the last column.}"

# Arg: glob pattern (works with *_split4_* loop)
PATTERN="${1:-$JOB_DIR/evolution_config_seed_1_generational_*}"

# Resolve to absolute search path
if [[ "$PATTERN" = /* || "$PATTERN" == "$BASE_DIR"* ]]; then
  SEARCH_PATTERN="$PATTERN"
else
  SEARCH_PATTERN="$BASE_DIR/${PATTERN#./}"
fi

mapfile -t JOB_FILES < <(compgen -G "$SEARCH_PATTERN" || true)
((${#JOB_FILES[@]})) || { echo "No jobs matched pattern: $PATTERN" >&2; exit 1; }

echo "Queueing ${#JOB_FILES[@]} jobs for $PROJECT; budget=$BUDGET; partition=$PARTITION"
for fn in "${JOB_FILES[@]}"; do
  echo "$fn"
  sbatch \
    -D "$BASE_DIR" \
    -A "$BUDGET" \
    --partition="$PARTITION" \
    --export=ALL \
    --time="$TIME_LIMIT" \
    --ntasks=1 \
    --gpus-per-task=4 \
    "$SCRIPT_PATH" \
    "$fn" \
    4
done

