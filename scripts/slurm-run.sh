#!/usr/bin/env sh
# SLURM wrapper for running experiments on JUWELS BOOSTER
#
# Intended to be run as:
# $ srun -A hai_XXXX --ntasks 4 --gpus-per-task=4 ~/einsearch/scripts/slurm-run.sh configs/einspace/evolution_test/addnist/evolution.yaml
set -e

# Configuration variables
CONFIG_FILE="$1"
WD=$(realpath $(dirname "$0"))
VENV_DIR="$WD/venv"
SCRIPT_PATH="$WD/run.sh"
GPU_COUNT=4

# Check for set config
if [ "$#" -ne 1 ]; then
    echo "$(basename """$0""") CONFIG_FILE" >&2
    exit 1
fi

# Activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "venv not found :("
    exit 1
fi
. "$VENV_DIR/bin/activate"

for gpu_id in $(seq 0 "$((GPU_COUNT - 1))"); do
    "$SCRIPT_PATH" "$CONFIG_FILE" "$gpu_id" &
done
wait
