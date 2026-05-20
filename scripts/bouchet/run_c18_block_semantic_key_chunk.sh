#!/usr/bin/env bash
set -euo pipefail

# Run one c18 block semantic-value key chunk from a manifest.
#
# Required:
#   MANIFEST=/path/to/block_semantic_key_manifest.json
# Optional:
#   TASK_ID defaults to SLURM_ARRAY_TASK_ID, then 0
#   DERIVATIVE_THREADS defaults to 7
#   BETA_CHUNK_SIZE defaults to 2
#   MAX_CHUNK_TERMS defaults to 200000
#   RECOMPUTE=1 forces recomputation of existing complete chunks
#   PYTHON_BIN defaults to python
#   PYTHON_MODULE optionally names the Yale module to load before Python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${MANIFEST:?set MANIFEST to the block semantic key manifest path}"
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
BETA_CHUNK_SIZE="${BETA_CHUNK_SIZE:-2}"
MAX_CHUNK_TERMS="${MAX_CHUNK_TERMS:-200000}"
RECOMPUTE="${RECOMPUTE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_MODULE="${PYTHON_MODULE:-}"

if [[ -n "${PYTHON_MODULE}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
  fi
  module load "${PYTHON_MODULE}"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RANK7_JK_DERIVATIVE_THREADS="${DERIVATIVE_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

args=(
  -m rank7_jk.c18_block_semantic_table run-key-manifest
  "${MANIFEST}"
  --task-id "${TASK_ID}"
  --beta-chunk-size "${BETA_CHUNK_SIZE}"
  --max-chunk-terms "${MAX_CHUNK_TERMS}"
)
if [[ "${RECOMPUTE}" == "1" ]]; then
  args+=(--recompute)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${args[@]}"
