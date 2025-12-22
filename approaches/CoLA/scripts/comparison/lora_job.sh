#!/bin/bash
#SBATCH --job-name=lora-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

set +u
if [[ -n "${CONDA_BASE:-}" && -n "${CONDA_ENV:-}" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
set -u

export WANDB_PROJECT="${WANDB_PROJECT:-htyllm-adapter-lpr}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-lpr-ablation}"
export WANDB_NAME="${WANDB_NAME:-lora-baseline}"
export WANDB_TAGS="${WANDB_TAGS:-comparison,lora,baseline}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
mkdir -p "${OUTPUT_DIR}"

# Avoid NFS-backed caches (can cause slowdowns/hangs on some clusters).
CACHE_ROOT="${CACHE_ROOT:-${TMPDIR:-/tmp}/${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/.triton/autotune}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME}/torch_extensions}"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

DATASET_DIR="${DATASET_DIR:-./LLaMA-Factory/data}"
TOKENIZED_PATH="${TOKENIZED_PATH:?TOKENIZED_PATH not set}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
DATASET_NAME="${DATASET_NAME:-c4}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-${DATASET_NAME}}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"

BF16="${BF16:-True}"
FP16="${FP16:-False}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-false}"

TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_LR="${LEARNING_RATE:-2e-4}"
TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
EVAL_BS="${PER_DEVICE_EVAL_BATCH_SIZE:-${TRAIN_BS}}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
ADDITIONAL_TARGET="${ADDITIONAL_TARGET:-}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"

RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))
RUN_NAME="${WANDB_NAME}_ep${TRAIN_EPOCHS}_bs${TRAIN_BS}x${GRAD_ACC}_lr${TRAIN_LR}_${RANDOM_ID}"
export WANDB_NAME="${RUN_NAME}"

echo "[INFO] Running LoRA baseline training into ${OUTPUT_DIR}"

ACCELERATE_CMD=()
ENTRYPOINT=()
if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  # Deterministic single-node rendezvous (avoid probe-then-use races with torch elastic).
  MASTER_ADDR="127.0.0.1"
  MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000) ))}"
  export MASTER_ADDR MASTER_PORT
  echo "[INFO] accelerate rendezvous MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
  ACCELERATE_CMD=(
    accelerate launch
    --config_file "${ACCELERATE_CONFIG_FILE}"
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT}"
    --module
  )
  # IMPORTANT: do not call `llamafactory-cli train` under accelerate, because LF will torchrun again.
  ENTRYPOINT=(llamafactory.launcher)
fi

if [[ "${#ENTRYPOINT[@]}" -eq 0 ]]; then
  LLAMAFATORY_CLI="$(command -v llamafactory-cli || true)"
  if [[ -z "${LLAMAFATORY_CLI}" ]]; then
    echo "[ERROR] llamafactory-cli not found in PATH" >&2
    exit 1
  fi
  ENTRYPOINT=("${LLAMAFATORY_CLI}" train)
fi

AUTO_FIND_BATCH_SIZE_FLAG=()
if [[ "${AUTO_FIND_BATCH_SIZE}" == "true" || "${AUTO_FIND_BATCH_SIZE}" == "True" || "${AUTO_FIND_BATCH_SIZE}" == "1" ]]; then
  AUTO_FIND_BATCH_SIZE_FLAG=(--auto_find_batch_size)
fi

"${ACCELERATE_CMD[@]}" "${ENTRYPOINT[@]}" \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET_NAME}" \
  --eval_dataset "${EVAL_DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type lora \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --run_name "${RUN_NAME}" \
  --num_train_epochs "${TRAIN_EPOCHS}" \
  --learning_rate "${TRAIN_LR}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --eval_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-2000}" \
  --save_steps "${SAVE_STEPS:-2000}" \
  --max_steps "${MAX_STEPS:-0}" \
  --disable_gradient_checkpointing "${DISABLE_GRADIENT_CHECKPOINTING:-True}" \
  --flash_attn "${FLASH_ATTN}" \
  --bf16 "${BF16:-False}" \
  --fp16 "${FP16:-True}" \
  "${AUTO_FIND_BATCH_SIZE_FLAG[@]}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --preprocessing_batch_size "${PREPROCESSING_BATCH_SIZE}" \
  --lora_rank "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target "${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}" \
  ${ADDITIONAL_TARGET:+--additional_target "${ADDITIONAL_TARGET}"} \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] LoRA baseline job completed"
touch "${OUTPUT_DIR}/.training_complete"
