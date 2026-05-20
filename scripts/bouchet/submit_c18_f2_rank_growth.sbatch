#!/usr/bin/env bash
#SBATCH --job-name=r7-c18-f2
#SBATCH --output=logs/c18_f2_%j.out
#SBATCH --error=logs/c18_f2_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Submit with, for example:
#   sbatch --time=04:00:00 \
#     --export=ALL,REPO_ROOT=$PWD,PRIME=1009,STOP_RANK=8,OUTPUT=$PWD/results/c18_f2_rank_growth/bouchet_p1009_rank8.json \
#     scripts/bouchet/submit_c18_f2_rank_growth.sbatch
#
# For the full p=101 run, use --time=1-00:00:00 and STOP_RANK=308.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
export PYTHON_MODULE

mkdir -p logs
bash "${REPO_ROOT}/scripts/bouchet/run_c18_f2_rank_growth.sh"
