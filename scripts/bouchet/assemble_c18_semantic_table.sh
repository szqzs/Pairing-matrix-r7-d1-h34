#!/usr/bin/env bash
set -euo pipefail

# Merge semantic key chunks into a value table, then assemble the matrix rank.
#
# Required:
#   MANIFEST=/path/to/semantic_key_manifest.json
# Optional:
#   TABLE defaults to results/c18_semantic_value_table.json.gz
#   OUTPUT defaults to results/c18_semantic_rank.json
#   LEFT_NULLSPACE=1 includes a left-nullspace basis
#   ALLOW_MISSING=1 permits partial table merges
#   PYTHON_BIN defaults to python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${MANIFEST:?set MANIFEST to the semantic key manifest path}"
TABLE="${TABLE:-results/c18_semantic_value_table.json.gz}"
OUTPUT="${OUTPUT:-results/c18_semantic_rank.json}"
LEFT_NULLSPACE="${LEFT_NULLSPACE:-1}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

merge_args=(
  -m rank7_jk.c18_semantic_table merge-key-manifest
  "${MANIFEST}"
  --output "${TABLE}"
)
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  merge_args+=(--allow-missing)
fi

assemble_args=(
  -m rank7_jk.c18_semantic_table assemble-rank
  "${TABLE}"
  --output "${OUTPUT}"
)
if [[ "${LEFT_NULLSPACE}" == "1" ]]; then
  assemble_args+=(--left-nullspace)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${merge_args[@]}"
"${PYTHON_BIN}" "${assemble_args[@]}"
