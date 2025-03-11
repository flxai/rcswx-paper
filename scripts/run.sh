#!/usr/bin/env bash
set -e

# take two input arguments, config file and the GPU id
CONFIG_FILE=$1
GPU_ID=$2
WD=$(realpath $(dirname "$0"))

# Check for set config and GPU id
if [ "$#" -ne 2 ]; then
    echo "$(basename """$1""") CONFIG_FILE GPU_ID" >&2
    exit 1
fi

# create log file from the config file
# e.g. configs/a/b/c/d.yaml -> logs/a/b/c/d.txt
LOG_FILE="logs/${CONFIG_FILE:8:-5}.txt"

# Create the log file, and its parent directories
mkdir -p $(dirname $LOG_FILE)
touch $LOG_FILE

# Run the script
python main.py --config $CONFIG_FILE --device cuda:$GPU_ID | tee $LOG_FILE
python test.py --config $CONFIG_FILE --device cuda:$GPU_ID | tee -a $LOG_FILE
python plot.py --config $CONFIG_FILE --device cuda:$GPU_ID
