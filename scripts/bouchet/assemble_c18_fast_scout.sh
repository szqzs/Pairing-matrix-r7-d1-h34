#!/usr/bin/env bash
set -euo pipefail

# Merge one c18 fast-scout manifest into a value table, then assemble rank.
#
# Required:
#   MANIFEST=/path/to/manifest.json
# Optional:
#   TABLE defaults to <manifest-dir>/value_table.json.gz
#   OUTPUT defaults to <manifest-dir>/rank.json
#   MAX_DEPENDENT_COLUMNS defaults to 32
#   NO_PLATEAU_STOP=1 computes the exact rank over all selected columns
#   LEFT_NULLSPACE=1 includes a left-nullspace basis
#   STORE_MATRIX=1 includes the assembled matrix
#   ALLOW_MISSING=1 permits partial table merges
#   PYTHON_BIN defaults to python
#   PYTHON_MODULE optionally names the Yale module to load before Python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${MANIFEST:?set MANIFEST to the fast-scout manifest path}"
MANIFEST_DIR="$(cd "$(dirname "${MANIFEST}")" && pwd)"
TABLE="${TABLE:-${MANIFEST_DIR}/value_table.json.gz}"
OUTPUT="${OUTPUT:-${MANIFEST_DIR}/rank.json}"
MAX_DEPENDENT_COLUMNS="${MAX_DEPENDENT_COLUMNS:-32}"
NO_PLATEAU_STOP="${NO_PLATEAU_STOP:-0}"
LEFT_NULLSPACE="${LEFT_NULLSPACE:-0}"
STORE_MATRIX="${STORE_MATRIX:-0}"
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
  -m rank7_jk_fast.semantic_pipeline merge
  "${MANIFEST}"
  --output "${TABLE}"
)
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  merge_args+=(--allow-missing)
fi

rank_args=(
  -m rank7_jk_fast.semantic_pipeline rank
  "${TABLE}"
  --output "${OUTPUT}"
)
if [[ "${NO_PLATEAU_STOP}" == "1" ]]; then
  rank_args+=(--no-plateau-stop)
else
  rank_args+=(--max-dependent-columns "${MAX_DEPENDENT_COLUMNS}")
fi
if [[ "${LEFT_NULLSPACE}" == "1" ]]; then
  rank_args+=(--compute-left-nullspace)
fi
if [[ "${STORE_MATRIX}" == "1" ]]; then
  rank_args+=(--store-matrix)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${merge_args[@]}"
"${PYTHON_BIN}" "${rank_args[@]}"
