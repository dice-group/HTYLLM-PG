#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export RUN_TRAIN_SMOKE=1
export RUN_LM_EVAL_SMOKE="${RUN_LM_EVAL_SMOKE:-1}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}"
export MODEL_NAME_OR_PATH
export SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/acl_smoke}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LM_EVAL_WANDB_MODE="${LM_EVAL_WANDB_MODE:-online}"
export SMOKE_DATASET_DATA_FILES="${SMOKE_DATASET_DATA_FILES:-${REPO_ROOT}/LLaMA-Factory/data/c4_demo.jsonl}"
export SMOKE_DATASET_DEFAULT_LANGUAGE="${SMOKE_DATASET_DEFAULT_LANGUAGE:-aeb_Arab}"
export SMOKE_SAVE_STEPS="${SMOKE_SAVE_STEPS:-1}"

# Example longer run with eval enabled:
# WANDB_API_KEY=your_key \
# LM_EVAL_BIN=lm_eval \
# RUN_LM_EVAL_SMOKE=1 \
# SMOKE_TRAIN_STEPS=20 \
# SMOKE_SAVE_STEPS=5 \
# SMOKE_BATCH_SIZE=1 \
# SMOKE_GRAD_ACCUM=1 \
# SMOKE_LOGGING_STEPS=5 \
# bash ./scripts/tests/run_acl_training_smoke.sh

"${PYTHON_BIN}" -m pytest "${REPO_ROOT}/tests/integration" ${EXTRA_PYTEST_ARGS:-}
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/tests/summary_acl_smoke.py" --root "${SMOKE_OUTPUT_ROOT}"

if [[ "${ENABLE_LM_EVAL_LISTENER:-1}" == "1" ]]; then
  LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
  if ! command -v "${LM_EVAL_BIN}" >/dev/null 2>&1; then
    echo "[WARN] lm_eval not found; skipping checkpoint eval."
    exit 0
  fi

  LM_EVAL_TASKS=${LM_EVAL_TASKS:-"belebele_zsm_Latn,belebele_zul_Latn,xnli"}
  LM_EVAL_EXTRA_ARGS=${LM_EVAL_EXTRA_ARGS:-"--limit 10"}
  LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-auto}
  LM_EVAL_WANDB_PROJECT=${LM_EVAL_WANDB_PROJECT:-acl_smoke_eval_debug}
  LM_EVAL_WANDB_PREFIX=${LM_EVAL_WANDB_PREFIX:-acl_smoke}
  LM_EVAL_WANDB_MODE=${LM_EVAL_WANDB_MODE:-online}

  for run_dir in "${SMOKE_OUTPUT_ROOT}"/*_*; do
    [[ -d "${run_dir}" ]] || continue
    "${REPO_ROOT}/scripts/tests/checkpoint_listener_local.sh" \
      --watch-dir "${run_dir}" \
      --eval-script "${REPO_ROOT}/scripts/tests/lm_eval_checkpoint_local.sh" \
      --tasks "${LM_EVAL_TASKS}" \
      --batch-size "${LM_EVAL_BATCH_SIZE}" \
      --wandb-project "${LM_EVAL_WANDB_PROJECT}" \
      --wandb-prefix "${LM_EVAL_WANDB_PREFIX}" \
      --wandb-mode "${LM_EVAL_WANDB_MODE}" \
      --extra-args "${LM_EVAL_EXTRA_ARGS}" \
      --once
  done
fi
