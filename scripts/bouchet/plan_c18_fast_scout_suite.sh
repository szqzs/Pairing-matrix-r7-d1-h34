#!/usr/bin/env bash
set -euo pipefail

# Plan the default c18 fast-scout semantic manifests.
#
# Optional:
#   OUTPUT_ROOT defaults to results/fast_c18_scouts/p${PRIME}
#   PRIME defaults to 101
#   METHOD defaults to semantic-batched
#   CHUNK_SIZE defaults to 100
#   PYTHON_BIN defaults to python
#   PYTHON_MODULE optionally names the Yale module to load before Python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PRIME="${PRIME:-101}"
METHOD="${METHOD:-semantic-batched}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fast_c18_scouts/p${PRIME}}"
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

mkdir -p "${OUTPUT_ROOT}" logs
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m rank7_jk_fast.semantic_pipeline plan-suite \
  --prime "${PRIME}" \
  --method "${METHOD}" \
  --chunk-size "${CHUNK_SIZE}" \
  --output-root "${OUTPUT_ROOT}"
