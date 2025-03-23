#!/usr/bin/env sh
# SLURM wrapper for running experiments on JUWELS BOOSTER
#
# Intended to be run as:
# $ sbatch -D ./scripts -A hai_XXXX --time 24:00:00 --ntasks 1 --gpus-per-task 4 ./scripts/slurm-run.sh ./configs/einspace.lst 4
set -e

# Check for set config
if [ "$#" -ne 2 ]; then
	echo "$(basename """$0""") BATCH_FILE GPU_COUNT" >&2
	exit 1
fi

# Configuration variables
BATCH_FILE="$1"
GPU_COUNT="$2"
WD="$PWD"
SCRIPT_PATH="$WD/run.sh"

# Set custom log files for SLURM
#SBATCH --output=${WD}/../logs/slurm/slurm-%j.out
#SBATCH --error=${WD}/../logs/slurm/slurm-%j.err

# Load JUWELS modules
while read -r mod; do
	module load "$mod"
done < "$WD/juwels_modules.txt"

# Parallelize via GNU Parallel
paste -d ' ' "$BATCH_FILE" | parallel --jobs "$GPU_COUNT" --colsep ';' "$SCRIPT_PATH" "../{1}" "{%}" "$SLURM_JOB_ID" {2}
