#!/usr/bin/env bash
set -euo pipefail

# Run one c18-even worker chunk from a manifest.
#
# Required:
#   MANIFEST=/path/to/manifest.json
# Optional:
#   TASK_ID defaults to SLURM_ARRAY_TASK_ID, then 0
#   DERIVATIVE_THREADS defaults to 7
#   PYTHON_BIN defaults to python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${MANIFEST:?set MANIFEST to the chunk manifest path}"
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-7}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RANK7_JK_DERIVATIVE_THREADS="${DERIVATIVE_THREADS}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m rank7_jk.c18_even_worker run-manifest \
  "${MANIFEST}" \
  --task-id "${TASK_ID}"
