#!/usr/bin/env bash
set -e

# take input arguments, path to config file first, then GPU id, then all other (optional) parameters
GPU_ID=$1
shift
CONFIG_FILE=$1
shift
ARGUMENTS=$@
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

echo "Using additional parameters:" &> "$LOG_FILE"
echo "  $ARGUMENTS" &>> "$LOG_FILE"
echo &>> "$LOG_FILE"

# Run the script
python "$WD/../main.py" $ARGUMENTS --config $CONFIG_FILE --device cuda:$GPU_ID &>> $LOG_FILE
python "$WD/../test.py" $ARGUMENTS --config $CONFIG_FILE --device cuda:$GPU_ID &>> $LOG_FILE
python "$WD/../plot.py" $ARGUMENTS --config $CONFIG_FILE --device cuda:$GPU_ID
