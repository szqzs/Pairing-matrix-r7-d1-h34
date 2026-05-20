#!/usr/bin/env bash
#SBATCH --job-name=r7-c18-even
#SBATCH --output=logs/c18_even_%A_%a.out
#SBATCH --error=logs/c18_even_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

# Submit with, for example:
#   sbatch --array=0-53 \
#     --export=ALL,MANIFEST=/path/to/manifest.json,REPO_ROOT=/path/to/repo \
#     scripts/bouchet/submit_c18_even_array.sbatch

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MANIFEST="${MANIFEST:?submit with --export=ALL,MANIFEST=/path/to/manifest.json}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-${SLURM_CPUS_PER_TASK:-7}}"

mkdir -p logs
bash "${REPO_ROOT}/scripts/bouchet/run_c18_even_chunk.sh"
