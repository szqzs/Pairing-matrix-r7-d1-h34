#!/usr/bin/env bash
set -euo pipefail

# Merge all completed chunk outputs listed in a manifest.
#
# Required:
#   MANIFEST=/path/to/manifest.json
# Optional:
#   OUTPUT defaults to results/c18_even_merged.json
#   PYTHON_BIN defaults to python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${MANIFEST:?set MANIFEST to the chunk manifest path}"
OUTPUT="${OUTPUT:-results/c18_even_merged.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m rank7_jk.c18_even_worker merge-manifest \
  "${MANIFEST}" \
  --output "${OUTPUT}"
