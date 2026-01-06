#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SLURM_NNODES="${SLURM_NNODES:-1}"
export SLURM_NODEID="${SLURM_NODEID:-0}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-hf-internal-testing/tiny-random-LlamaForCausalLM}"
TOKENIZED_PATH="${TOKENIZED_PATH:-/data/project_data/moe_study/tokenized/preview_subset_tiny_llama}"
LANGUAGE_MAP="${LANGUAGE_MAP:-/upb/users/j/joeldag/profiles/unix/cs/HTYLLM-PG/approaches/CoLA/tools/two_stage_clustering/200_tier_language_groupings.json}"
LANGUAGE_COLUMN="${LANGUAGE_COLUMN:-language}"
DATASET_NAME="${DATASET_NAME:-identity}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/LLaMA-Factory/data}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-${DATASET_NAME}}"

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH is required." >&2
  exit 1
fi
if [[ -z "${TOKENIZED_PATH}" ]]; then
  echo "[ERROR] TOKENIZED_PATH is required." >&2
  exit 1
fi
if [[ ! -f "${LANGUAGE_MAP}" ]]; then
  echo "[ERROR] LANGUAGE_MAP not found at ${LANGUAGE_MAP}." >&2
  exit 1
fi

if [[ -z "${COLA_NUM_EXPERTS:-}" ]]; then
  COLA_NUM_EXPERTS=$(
    python3 - <<'PY' "${LANGUAGE_MAP}"
import json
import sys
path = sys.argv[1]
data = json.load(open(path))
print(len(data) if isinstance(data, dict) else len(data))
PY
  )
  export COLA_NUM_EXPERTS
fi

if [[ -z "${COLA_EXPERT_NUM_B:-}" ]]; then
  COLA_EXPERT_NUM_B=$(
    python3 - <<'PY' "${LANGUAGE_MAP}"
import json
import sys
path = sys.argv[1]
data = json.load(open(path))
if not isinstance(data, dict):
    print("")
    raise SystemExit(0)
counts = []
for _, entry in sorted(data.items(), key=lambda kv: str(kv[0])):
    if not isinstance(entry, dict):
        counts.append(0)
        continue
    subgroups = entry.get("subgroups") or {}
    counts.append(len(subgroups) if isinstance(subgroups, dict) else 0)
print("" if all(c == 0 for c in counts) else ",".join(str(c) for c in counts))
PY
  )
  export COLA_EXPERT_NUM_B
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/local_cola_lpr_2gpu}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/cola_lpr_${RUN_TAG}}"

export OUTPUT_DIR
export MODEL_NAME_OR_PATH
export TOKENIZED_PATH
export DATASET_NAME
export DATASET_DIR
export EVAL_DATASET_NAME
export LANGUAGE_MAP
export LANGUAGE_COLUMN

export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${REPO_ROOT}/LLaMA-Factory/examples/accelerate/fsdp_2gpu_config.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-local_cola_lpr}"
export FLASH_ATTN="${FLASH_ATTN:-disabled}"
export SAVE_STEPS="${SAVE_STEPS:-40}"
export EVAL_STEPS="${EVAL_STEPS:-50}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
export MAX_STEPS="${MAX_STEPS:-100}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export USE_COLA_EXPERTS="${USE_COLA_EXPERTS:-True}"
export COLA_NUM_A="${COLA_NUM_A:-1}"
export COLA_NUM_B="${COLA_NUM_B:-3}"
export COLA_TOP_K="${COLA_TOP_K:-1}"
export COLA_STRATEGY="${COLA_STRATEGY:-fully}"
export LANGUAGE_ROUTER_MODE="${LANGUAGE_ROUTER_MODE:-learned}"
export LANGUAGE_HEAD_ROUTER_MODE="${LANGUAGE_HEAD_ROUTER_MODE:-learned}"
export LANGUAGE_PRIOR_WEIGHT="${LANGUAGE_PRIOR_WEIGHT:-0.1}"
export LANGUAGE_BIAS_VALUE="${LANGUAGE_BIAS_VALUE:-0.0}"
export LANGUAGE_HEAD_BIAS_VALUE="${LANGUAGE_HEAD_BIAS_VALUE:-0.0}"
export LANGUAGE_GUIDANCE_SCOPE="${LANGUAGE_GUIDANCE_SCOPE:-all}"

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Launching local CoLA LPR (2 GPU FSDP) into ${OUTPUT_DIR}"
bash "${REPO_ROOT}/scripts/comparison/cola_lpr_job.sh"

if [[ "${MERGE_ADAPTER_SHARDS:-1}" == "1" ]]; then
  mapfile -t CKPTS < <(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  if [[ ${#CKPTS[@]} -gt 0 ]]; then
    echo "[INFO] Merging adapter shards for ${#CKPTS[@]} checkpoints..."
    for ckpt in "${CKPTS[@]}"; do
      if [[ -d "${ckpt}_adapter_sharded" && ! -f "${ckpt}_adapter/adapter_model.safetensors" ]]; then
        python3 "${REPO_ROOT}/scripts/merge_adapter_shards.py" \
          --adapter-sharded-dir "${ckpt}_adapter_sharded" \
          --output-dir "${ckpt}_adapter" \
          --base-model "${MODEL_NAME_OR_PATH}"
      fi
    done
  fi
fi

if [[ "${RUN_LM_EVAL:-1}" == "1" ]]; then
if [[ -f "/upb/users/j/joeldag/profiles/unix/cs/HTYLLM-PG/approaches/CoLA/configs/lm_eval_tasks.txt" ]]; then
  LM_EVAL_TASKS="${LM_EVAL_TASKS:-$(paste -sd, "/upb/users/j/joeldag/profiles/unix/cs/HTYLLM-PG/approaches/CoLA/configs/lm_eval_tasks.txt")}"
else
  LM_EVAL_TASKS="${LM_EVAL_TASKS:-belebele_zul_Latn}"
fi
  LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-1}"
  LM_EVAL_WANDB_PROJECT="${LM_EVAL_WANDB_PROJECT:-local_lm_eval}"
  LM_EVAL_WANDB_PREFIX="${LM_EVAL_WANDB_PREFIX:-cola_lpr}"
  LM_EVAL_WANDB_MODE="${LM_EVAL_WANDB_MODE:-disabled}"
  export LM_EVAL_LANG_MODE="${LM_EVAL_LANG_MODE:-both}"
  export LM_EVAL_TORCH_DTYPE="${LM_EVAL_TORCH_DTYPE:-bf16}"
  export LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-10}"

  "${REPO_ROOT}/scripts/tests/checkpoint_listener_local.sh" \
    --watch-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/lm_eval" \
    --eval-script "${REPO_ROOT}/scripts/tests/lm_eval_checkpoint_local.sh" \
    --tasks "${LM_EVAL_TASKS}" \
    --batch-size "${LM_EVAL_BATCH_SIZE}" \
    --wandb-project "${LM_EVAL_WANDB_PROJECT}" \
    --wandb-prefix "${LM_EVAL_WANDB_PREFIX}" \
    --wandb-mode "${LM_EVAL_WANDB_MODE}" \
    --once
fi

echo "[INFO] Local CoLA LPR run complete."
