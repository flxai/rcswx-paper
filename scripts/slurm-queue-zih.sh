#!/usr/bin/env bash
# Queue Slurm jobs. Use scripts/slurm-split.sh beforehand to split jobs into single files.
set -euo pipefail
shopt -s nullglob

BASE_DIR="${BASE_DIR:-$HOME/einsearch}"
JOB_DIR="$BASE_DIR/jobs"
SCRIPT_PATH="$BASE_DIR/scripts/slurm-run.sh"

# Args:
#   $1: glob pattern of job files; default matches split1 jobs.
#   $2: job-name control. If unset -> auto from filename.
#       If "auto" -> basename (no ext). If contains "{}" or "%s" -> replaced by basename.
#       Else -> fixed job name for all submissions.
PATTERN="${1:-$JOB_DIR/evolution_config_*_split1_*}"
JOB_NAME_FMT="${2-}"

# Expand pattern robustly (handles no-match without failing).
mapfile -t JOB_FILES < <(compgen -G "$PATTERN" || true)
if ((${#JOB_FILES[@]} == 0)); then
  echo "No jobs matched pattern: $PATTERN" >&2
  exit 1
fi

echo "Queueing ${#JOB_FILES[@]} jobs matching: $PATTERN"
for fn in "${JOB_FILES[@]}"; do
  base="$(basename "$fn")"
  base="${base%.*}"

  if [[ -n "${JOB_NAME_FMT:-}" ]]; then
    if [[ "$JOB_NAME_FMT" == "auto" ]]; then
      jobname="$base"
    elif [[ "$JOB_NAME_FMT" == *'{}'* ]]; then
      jobname="${JOB_NAME_FMT//\{\}/$base}"
    elif [[ "$JOB_NAME_FMT" == *'%s'* ]]; then
      jobname="${JOB_NAME_FMT//%s/$base}"
    else
      jobname="$JOB_NAME_FMT"
    fi
  else
    jobname="$base"
  fi

  # Slurm JobName max length ~128 chars; truncate to be safe.
  jobname="${jobname:0:128}"

  echo "$fn  ->  job-name: $jobname"
  sbatch \
    -D "$BASE_DIR" \
    --export=ALL \
    --time=48:00:00 \
    --cpus-per-task=6 \
    --gres=gpu:1 \
    --mem=120G \
    --ntasks=1 \
    --nodes=1 \
    --job-name="$jobname" \
    "$SCRIPT_PATH" \
    "$fn" \
    1
done
