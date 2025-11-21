#!/usr/bin/env bash
#SBATCH --job-name=rcswx-distance
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G
#SBATCH --output=logs/distance-slurm/%j.out
#SBATCH --error=logs/distance-slurm/%j.err
#
# SLURM wrapper for rcswx-distance.py (one job per CSV)
#
# Example usage:
#   find jobs_lists_split -maxdepth 1 -type f -name 'compute_distance_cifar10_0_*' -print0 \
#   | xargs -0 -n1 -I{} sbatch --export=ALL,CSV="{}" scripts/rcswx-distance.sh

set -euo pipefail
mkdir -p logs/distance-slurm

# Change to repo' root dir
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
cd "${REPO_ROOT}"

# ZIH HPC's modules
echo "Loading modules..."
module load release/24.10 GCCcore/13.2.0 Python/3.12.3 Graphviz/8.1.0
echo "Loading venv..."
source "venv/bin/activate"

echo "Starting distance computations..."
NWORKERS=$(( SLURM_CPUS_PER_TASK * 10 ))
CSV="${CSV:?set CSV via --export=ALL,CSV=/path/to/file}"
srun --cpu-bind=cores \
  python -u scripts/rcswx-distance.py "$CSV" \
    --workers "${NWORKERS}" \
    --results-dir results/distance \
    --logs-dir logs/distance \
    --skip-existing
