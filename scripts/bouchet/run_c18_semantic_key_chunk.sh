#!/usr/bin/env bash
set -euo pipefail

# Run one c18 semantic-value key chunk from a manifest.
#
# Required:
#   MANIFEST=/path/to/semantic_key_manifest.json
# Optional:
#   TASK_ID defaults to SLURM_ARRAY_TASK_ID, then 0
#   DERIVATIVE_THREADS defaults to 7
#   BETA_CHUNK_SIZE defaults to 2
#   MAX_CHUNK_TERMS defaults to 200000
#   RECOMPUTE=1 forces recomputation of existing complete chunks
#   PYTHON_BIN defaults to python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${MANIFEST:?set MANIFEST to the semantic key manifest path}"
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-7}"
BETA_CHUNK_SIZE="${BETA_CHUNK_SIZE:-2}"
MAX_CHUNK_TERMS="${MAX_CHUNK_TERMS:-200000}"
RECOMPUTE="${RECOMPUTE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RANK7_JK_DERIVATIVE_THREADS="${DERIVATIVE_THREADS}"

args=(
  -m rank7_jk.c18_semantic_table run-key-manifest
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
