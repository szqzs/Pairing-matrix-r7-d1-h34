#!/usr/bin/env bash
#SBATCH --job-name=r7-c18-block-keys
#SBATCH --output=logs/c18_block_keys_%A_%a.out
#SBATCH --error=logs/c18_block_keys_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

# Submit with, for example:
#   sbatch --array=0-99 \
#     --export=ALL,MANIFEST=/path/to/block_semantic_key_manifest.json,REPO_ROOT=/path/to/repo \
#     scripts/bouchet/submit_c18_block_semantic_keys.sbatch

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MANIFEST="${MANIFEST:?submit with --export=ALL,MANIFEST=/path/to/manifest.json}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
BETA_CHUNK_SIZE="${BETA_CHUNK_SIZE:-2}"
MAX_CHUNK_TERMS="${MAX_CHUNK_TERMS:-200000}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
export PYTHON_MODULE

mkdir -p logs
bash "${REPO_ROOT}/scripts/bouchet/run_c18_block_semantic_key_chunk.sh"
