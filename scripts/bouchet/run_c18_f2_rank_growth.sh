#!/usr/bin/env bash
set -euo pipefail

# Run the high-f2 c18 rank-growth job.
#
# Required for normal use:
#   OUTPUT=/path/to/output.json
#
# Optional:
#   PRIME defaults to 101
#   STOP_RANK defaults to empty; TARGET_LEFT_NULLITY defaults to empty
#   CHECKPOINT defaults to OUTPUT with ".checkpoint.json"
#   RESUME_FROM defaults to CHECKPOINT if it exists
#   DERIVATIVE_THREADS defaults to SLURM_CPUS_PER_TASK, then 1
#   PYTHON_BIN defaults to python

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_MODULE="${PYTHON_MODULE:-}"
PRIME="${PRIME:-101}"
METHOD="${METHOD:-batched}"
ROW_KIND="${ROW_KIND:-all}"
ROW_ORDER="${ROW_ORDER:-defect-balanced}"
COLUMN_ORDER="${COLUMN_ORDER:-sequential}"
STOP_RANK="${STOP_RANK:-}"
TARGET_LEFT_NULLITY="${TARGET_LEFT_NULLITY:-}"
MAX_COLUMNS="${MAX_COLUMNS:-}"
MAX_SEMANTIC_KEYS="${MAX_SEMANTIC_KEYS:-}"
BETA_CHUNK_SIZE="${BETA_CHUNK_SIZE:-2}"
MAX_CHUNK_TERMS="${MAX_CHUNK_TERMS:-200000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
DERIVATIVE_THREADS="${DERIVATIVE_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

if [[ -n "${PYTHON_MODULE}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
  fi
  module load "${PYTHON_MODULE}"
fi

OUTPUT="${OUTPUT:?set OUTPUT to the rank-growth JSON output path}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT%.json}.checkpoint.json}"
RESUME_FROM="${RESUME_FROM:-}"
if [[ -z "${RESUME_FROM}" && -f "${CHECKPOINT}" ]]; then
  RESUME_FROM="${CHECKPOINT}"
fi

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${CHECKPOINT}")" logs

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RANK7_JK_DERIVATIVE_THREADS="${DERIVATIVE_THREADS}"

args=(
  -m rank7_jk.c18_f2_rank_growth
  --prime "${PRIME}"
  --method "${METHOD}"
  --row-kind "${ROW_KIND}"
  --row-order "${ROW_ORDER}"
  --column-order "${COLUMN_ORDER}"
  --checkpoint "${CHECKPOINT}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --output "${OUTPUT}"
)

if [[ -n "${STOP_RANK}" ]]; then
  args+=(--stop-rank "${STOP_RANK}")
elif [[ -n "${TARGET_LEFT_NULLITY}" ]]; then
  args+=(--target-left-nullity "${TARGET_LEFT_NULLITY}")
else
  args+=(--no-target-left-nullity)
fi

if [[ -n "${MAX_COLUMNS}" ]]; then
  args+=(--max-columns "${MAX_COLUMNS}")
fi
if [[ -n "${MAX_SEMANTIC_KEYS}" ]]; then
  args+=(--max-semantic-keys "${MAX_SEMANTIC_KEYS}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  args+=(--resume-from "${RESUME_FROM}")
fi

args+=(--beta-chunk-size "${BETA_CHUNK_SIZE}")
args+=(--max-chunk-terms "${MAX_CHUNK_TERMS}")

cd "${REPO_ROOT}"
echo "Running high-f2 rank growth in ${REPO_ROOT}"
echo "Output: ${OUTPUT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Prime: ${PRIME}"
echo "Stop rank: ${STOP_RANK:-<none>}; target left nullity: ${TARGET_LEFT_NULLITY:-<none>}"
echo "Derivative threads: ${DERIVATIVE_THREADS}"
"${PYTHON_BIN}" "${args[@]}"
