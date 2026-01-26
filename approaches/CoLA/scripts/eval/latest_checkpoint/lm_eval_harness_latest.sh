#!/bin/bash
#SBATCH --job-name=lastckp-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HARNESS_ROOT="${LM_EVAL_HARNESS_PATH:-${REPO_ROOT}/scripts/eval/lm-evaluation-harness}"
LLAMAFACTORY_SRC="${REPO_ROOT}/LLaMA-Factory/src"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH not set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
TASKS_FILE="${TASKS_FILE:-${REPO_ROOT}/configs/lm_eval_tasks.txt}"
BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LANG_MODE="${LM_EVAL_LANG_MODE:-both}"
LIMIT="${LM_EVAL_LIMIT:-}"
LOG_ROUTER_METRICS="${LM_EVAL_LOG_ROUTER_METRICS:-true}"
FORCE_DEVICE="${LM_EVAL_FORCE_DEVICE:-true}"
INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-}"
DEVICE="${LM_EVAL_DEVICE:-cuda}"

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Checkpoint dir not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}/adapter_config.json" && -f "${CHECKPOINT_PATH}_adapter/adapter_config.json" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_PATH}_adapter"
fi
if [[ ! -f "${CHECKPOINT_PATH}/adapter_config.json" ]]; then
  echo "[ERROR] adapter_config.json not found in ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TASKS_FILE}" ]]; then
  echo "[ERROR] tasks file not found: ${TASKS_FILE}" >&2
  exit 1
fi

# Resolve base model from adapter_config.json
BASE_MODEL=$(python3 - <<'PY' "${CHECKPOINT_PATH}/adapter_config.json"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("base_model_name_or_path", ""))
PY
)
if [[ -z "${BASE_MODEL}" ]]; then
  echo "[ERROR] base_model_name_or_path missing in adapter_config.json" >&2
  exit 1
fi

# Build tasks list from file
TASKS=$(grep -v '^[[:space:]]*#' "${TASKS_FILE}" | awk 'NF' | paste -sd, -)
if [[ -z "${TASKS}" ]]; then
  echo "[ERROR] no tasks found in ${TASKS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

set +u
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-${CONDA_ENV:-}}"
if [[ -n "${CONDA_BASE:-}" && -n "${EVAL_CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${EVAL_CONDA_ENV}"
fi
set -u

# Ensure local harness + local PEFT are used
export PYTHONPATH="${HARNESS_ROOT}:${LLAMAFACTORY_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

WANDB_PROJECT="${WANDB_PROJECT:-lm-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_ARGS="project=${WANDB_PROJECT}"
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=" ,entity=${WANDB_ENTITY}"
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  WANDB_ARGS+=" ,group=${WANDB_GROUP}"
fi
if [[ -n "${WANDB_NAME}" ]]; then
  WANDB_ARGS+=" ,name=${WANDB_NAME}"
fi
WANDB_ARGS+=" ,job_type=lm-eval"
WANDB_ARGS=$(echo "${WANDB_ARGS}" | tr -d ' ')

EXTRA_ARGS=()
if [[ -n "${INCLUDE_PATH}" ]]; then
  EXTRA_ARGS+=("--include-path" "${INCLUDE_PATH}")
fi
if [[ -n "${LIMIT}" ]]; then
  EXTRA_ARGS+=("--limit" "${LIMIT}")
fi
if [[ "${LOG_ROUTER_METRICS}" == "true" ]]; then
  EXTRA_ARGS+=("--log-router-metrics")
fi

LM_EVAL_FORCE_DEVICE="${FORCE_DEVICE}" \
python3 "${REPO_ROOT}/scripts/eval/lm_eval_language_ids.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --tokenizer "${BASE_MODEL}" \
  --tasks "${TASKS}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --mode "${LANG_MODE}" \
  --device "${DEVICE}" \
  --wandb-args "${WANDB_ARGS}" \
  "${EXTRA_ARGS[@]}"
