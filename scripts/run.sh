#!/usr/bin/env bash
set -e

# Check for set config, GPU and job id
if [ "$#" -le 2 ]; then
	echo "$(basename """$0""") CONFIG_FILE GPU_ID JOB_ID [CONFIG_OVERWRITES]" >&2
	exit 1
fi

# take input arguments, path to config file first, then GPU id, then all other (optional) parameters
CONFIG_FILE=$1
shift
GPU_ID=$1
shift
JOB_ID=$1
shift
ARGUMENTS=$@
WD=$(realpath $(dirname "$0"))

# Activate virtual environment
WD=$(realpath $(dirname "$0"))
VENV_DIR="$WD/venv"
if [ ! -d "$VENV_DIR" ]; then
	echo "venv not found. Creating one..."
	python -m venv venv
	. "$VENV_DIR/bin/activate"
	pip install --no-input einops graphviz positional_encodings psutil pydot pygraphviz pympler rich termcolor tqdm typing_extensions
else
	. "$VENV_DIR/bin/activate"
fi

# Limit PyTorch to single GPU
export CUDA_VISIBLE_DEVICES="$((GPU_ID - 1))"

# Required for reproducibility in PyTorch/CuBLAS
# https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Create log file from the config file
# e.g. configs/a/b/c/d.yaml -> logs/a/b/c/d.txt
LOG_FILE="$WD/../logs/${CONFIG_FILE:8:-5}_${JOB_ID}_${GPU_ID}.txt"

# Create log dir & file
mkdir -p """$(dirname "$LOG_FILE")"""

touch $LOG_FILE

echo "Log file for einsearch experiment" &> "$LOG_FILE"
echo "Using parameters:" &>> "$LOG_FILE"
echo "  config file: $CONFIG_FILE" &>> "$LOG_FILE"
echo "       GPU ID: $GPU_ID" &>> "$LOG_FILE"
echo "       JOB ID: $JOB_ID" &>> "$LOG_FILE"
echo "Using additional parameters:" &>> "$LOG_FILE"
echo "  $ARGUMENTS" &>> "$LOG_FILE"
echo &>> "$LOG_FILE"

# Run the script
python "$WD/../main.py" $ARGUMENTS --config "$CONFIG_FILE" &>> "$LOG_FILE"
python "$WD/../test.py" $ARGUMENTS --config "$CONFIG_FILE" &>> "$LOG_FILE"
# python "$WD/../plot.py" $ARGUMENTS --config "$CONFIG_FILE"
