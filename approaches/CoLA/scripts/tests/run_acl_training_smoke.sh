#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export RUN_TRAIN_SMOKE=1

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
export MODEL_NAME_OR_PATH
export SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/acl_smoke}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PYTHON_BIN}" -m pytest "${REPO_ROOT}/tests/integration" ${EXTRA_PYTEST_ARGS:-}
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/tests/summary_acl_smoke.py" --root "${SMOKE_OUTPUT_ROOT}"
