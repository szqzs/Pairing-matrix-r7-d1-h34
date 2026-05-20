#!/usr/bin/env bash
#SBATCH --job-name=r7-c18-fast-scout
#SBATCH --output=logs/c18_fast_scout_%A_%a.out
#SBATCH --error=logs/c18_fast_scout_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

# Submit with, for example:
#   sbatch --array=0-80%16 \
#     --export=ALL,MANIFEST=$PWD/results/fast_c18_scouts/p101/even_b_pair_64/manifest.json,REPO_ROOT=$PWD \
#     scripts/bouchet/submit_c18_fast_scout.sbatch

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MANIFEST="${MANIFEST:?submit with --export=ALL,MANIFEST=/path/to/manifest.json}"
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
BETA_CHUNK_SIZE="${BETA_CHUNK_SIZE:-2}"
MAX_CHUNK_TERMS="${MAX_CHUNK_TERMS:-200000}"
RECOMPUTE="${RECOMPUTE:-0}"
RESIDUE_BACKEND="${RESIDUE_BACKEND:-}"
PRODUCT_PROFILE="${PRODUCT_PROFILE:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

if [[ -n "${PYTHON_MODULE}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
  fi
  module load "${PYTHON_MODULE}"
fi

mkdir -p logs

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

args=(
  -m rank7_jk_fast.semantic_pipeline run-chunk
  "${MANIFEST}"
  --task-id "${TASK_ID}"
  --derivative-threads "${DERIVATIVE_THREADS}"
  --beta-chunk-size "${BETA_CHUNK_SIZE}"
  --max-chunk-terms "${MAX_CHUNK_TERMS}"
)
if [[ "${RECOMPUTE}" == "1" ]]; then
  args+=(--recompute)
fi
if [[ -n "${RESIDUE_BACKEND}" ]]; then
  args+=(--residue-backend "${RESIDUE_BACKEND}")
fi
if [[ -n "${PRODUCT_PROFILE}" ]]; then
  args+=(--product-profile "${PRODUCT_PROFILE}")
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${args[@]}"
