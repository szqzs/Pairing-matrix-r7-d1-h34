#!/usr/bin/env bash
set -euo pipefail

# Merge c18 block semantic key chunks into a value table, then assemble rank.
#
# Required:
#   MANIFEST=/path/to/block_semantic_key_manifest.json
# Optional:
#   TABLE defaults to results/c18_block_semantic_value_table.json.gz
#   OUTPUT defaults to results/c18_block_semantic_rank.json
#   LEFT_NULLSPACE=1 includes a left-nullspace basis
#   ALLOW_MISSING=1 permits partial table merges
#   PYTHON_BIN defaults to python
#   PYTHON_MODULE optionally names the Yale module to load before Python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${MANIFEST:?set MANIFEST to the block semantic key manifest path}"
TABLE="${TABLE:-results/c18_block_semantic_value_table.json.gz}"
OUTPUT="${OUTPUT:-results/c18_block_semantic_rank.json}"
LEFT_NULLSPACE="${LEFT_NULLSPACE:-1}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
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

merge_args=(
  -m rank7_jk.c18_block_semantic_table merge-key-manifest
  "${MANIFEST}"
  --output "${TABLE}"
)
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  merge_args+=(--allow-missing)
fi

assemble_args=(
  -m rank7_jk.c18_block_semantic_table assemble-rank
  "${TABLE}"
  --output "${OUTPUT}"
)
if [[ "${LEFT_NULLSPACE}" == "1" ]]; then
  assemble_args+=(--left-nullspace)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${merge_args[@]}"
"${PYTHON_BIN}" "${assemble_args[@]}"
