#!/usr/bin/env sh
# SLURM wrapper for running experiments on JUWELS BOOSTER
#
# Intended to be run as:
# $ srun -A hai_XXXX --ntasks 4 --gpus-per-task=4 scripts/slurm-run.sh configs/einspace.lst 4
set -e

# Configuration variables
BATCH_FILE="$1"
GPU_COUNT="$2"
WD=$(realpath $(dirname "$0"))
VENV_DIR="$WD/venv"
SCRIPT_PATH="$WD/run.sh"

# Check for set config
if [ "$#" -ne 2 ]; then
	echo "$(basename """$0""") BATCH_FILE GPU_COUNT" >&2
	exit 1
fi

# Activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
	echo "venv not found :("
	exit 1
fi
. "$VENV_DIR/bin/activate"

# Parallelize via GNU Parallel
paste -d ' ' "$BATCH_FILE" | parallel --jobs "$GPU_COUNT" --colsep ';' "$SCRIPT_PATH" "{1}" "{%}" {2}
