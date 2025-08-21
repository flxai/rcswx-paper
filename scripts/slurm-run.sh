#!/usr/bin/env sh
# SLURM wrapper for running experiments on JUWELS BOOSTER
#
# Intended to be run as:
# $ sbatch -D /einsearch --export ALL -A hai_XXXX --time 24:00:00 --ntasks 1 --gpus-per-task 4 /einsearch/scripts/slurm-run.sh /einsearch/configs/einspace.lst 4
#
# Set custom log files for SLURM
#SBATCH --output=logs/slurm/slurm-%j.out
#SBATCH --error=logs/slurm/slurm-%j.err
set -e

# Check for set config
if [ "$#" -ne 2 ]; then
	echo "$(basename """$0""") BATCH_FILE GPU_COUNT" >&2
	exit 1
fi

# Configuration variables
BATCH_FILE="$1"
GPU_COUNT="$2"
REPO_ROOT="$PWD"
SCRIPT_PATH="$REPO_ROOT/scripts/run.sh"

# ZIH Dresden
module load release/24.10 GCCcore/13.2.0 Python/3.12.3 Graphviz/8.1.0

# DEBUG Just run first single job instead
# IFS=';' read -r cfg options <<< $(head -1 "$BATCH_FILE")
# "$SCRIPT_PATH" "$cfg" "0" "$SLURM_JOB_ID" $options

# Parallelize via GNU Parallel
paste -d ' ' "$BATCH_FILE" | parallel --jobs "$GPU_COUNT" --colsep ';' "$SCRIPT_PATH" "{1}" "{%}" "$SLURM_JOB_ID" {2}
