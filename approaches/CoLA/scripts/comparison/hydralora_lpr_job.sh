#!/bin/bash
#SBATCH --job-name=lpr-hydra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/lpr_ablation/hydra_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${MODULE_INIT:-}" ]]; then
  eval "${MODULE_INIT}"
fi

if [[ -n "${CONDA_BASE:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-htyllm-adapter-lpr}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-lpr-ablation}"
export WANDB_NAME="${WANDB_NAME:-hydralora-lpr}"
export WANDB_TAGS="${WANDB_TAGS:-comparison,hydralora,lpr}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
mkdir -p "${OUTPUT_DIR}"

DATASET_DIR="${DATASET_DIR:-./LLaMA-Factory/data}"
TOKENIZED_PATH="${TOKENIZED_PATH:?TOKENIZED_PATH not set}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH not set}"
DATASET_NAME="${DATASET_NAME:-c4}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-${DATASET_NAME}}"

LANGUAGE_MAP="${LANGUAGE_MAP:?LANGUAGE_MAP not set}"
LANGUAGE_COLUMN="${LANGUAGE_COLUMN:-language}"
LANGUAGE_ROUTER_MODE="${LANGUAGE_ROUTER_MODE:-learned}"
LANGUAGE_PRIOR_WEIGHT="${LANGUAGE_PRIOR_WEIGHT:-0.0}"
LANGUAGE_BIAS_VALUE="${LANGUAGE_BIAS_VALUE:-0.0}"
LANGUAGE_HEAD_ROUTER_MODE="${LANGUAGE_HEAD_ROUTER_MODE:-${LANGUAGE_ROUTER_MODE}}"
LANGUAGE_HEAD_BIAS_VALUE="${LANGUAGE_HEAD_BIAS_VALUE:-${LANGUAGE_BIAS_VALUE}}"
LANGUAGE_GUIDANCE_SCOPE="${LANGUAGE_GUIDANCE_SCOPE:-none}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"

USE_HYDRALORA_EXPERTS="${USE_HYDRALORA_EXPERTS:-False}"
HYDRALORA_NUM_EXPERTS="${HYDRALORA_NUM_EXPERTS:-1}"
HYDRALORA_TOP_K="${HYDRALORA_TOP_K:-1}"

TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_LR="${LEARNING_RATE:-2e-4}"
TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
EVAL_BS="${PER_DEVICE_EVAL_BATCH_SIZE:-${TRAIN_BS}}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_NUM="${LORA_NUM:-4}"
ADDITIONAL_TARGET="${ADDITIONAL_TARGET:-}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"

RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))
RUN_NAME="${WANDB_NAME}_ep${TRAIN_EPOCHS}_bs${TRAIN_BS}x${GRAD_ACC}_lr${TRAIN_LR}_heads${LORA_NUM}_${RANDOM_ID}"
export WANDB_NAME="${RUN_NAME}"

# Explicitly log routing params to W&B config
export WANDB_CONFIG_JSON=$(cat <<EOF
{
  "language_router_mode": "${LANGUAGE_ROUTER_MODE}",
  "language_head_router_mode": "${LANGUAGE_HEAD_ROUTER_MODE}",
  "language_prior_weight": ${LANGUAGE_PRIOR_WEIGHT},
  "language_bias_value": ${LANGUAGE_BIAS_VALUE},
  "language_head_bias_value": ${LANGUAGE_HEAD_BIAS_VALUE},
  "language_guidance_scope": "${LANGUAGE_GUIDANCE_SCOPE}",
  "use_hydralora_experts": ${USE_HYDRALORA_EXPERTS},
  "hydralora_num_experts": ${HYDRALORA_NUM_EXPERTS},
  "hydralora_top_k": ${HYDRALORA_TOP_K},
  "lora_num": ${LORA_NUM}
}
EOF
)

echo "[INFO] Running HydraLoRA LPR training into ${OUTPUT_DIR}"

python "${REPO_ROOT}/scripts/comparison/router_setup.py" --type hydra

ACCELERATE_CMD=()
if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  ACCELERATE_CMD=(accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}")
fi

LLAMAFATORY_CLI="$(command -v llamafactory-cli || true)"
if [[ -z "${LLAMAFATORY_CLI}" ]]; then
  echo "[ERROR] llamafactory-cli not found in PATH" >&2
  exit 1
fi

"${ACCELERATE_CMD[@]}" "${LLAMAFATORY_CLI}" train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET_NAME}" \
  --eval_dataset "${EVAL_DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type hydralora \
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
  --evaluation_strategy "${EVAL_STRATEGY:-steps}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --max_steps "${MAX_STEPS:-0}" \
  --disable_gradient_checkpointing "${DISABLE_GRADIENT_CHECKPOINTING:-True}" \
  --bf16 "${BF16:-False}" \
  --fp16 "${FP16:-True}" \
  --seed "${SEED:-42}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --preprocessing_batch_size "${PREPROCESSING_BATCH_SIZE}" \
  --lora_rank "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_num "${LORA_NUM}" \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}" \
  --language_router_mode "${LANGUAGE_ROUTER_MODE}" \
  --language_head_router_mode "${LANGUAGE_HEAD_ROUTER_MODE}" \
  --language_prior_weight "${LANGUAGE_PRIOR_WEIGHT}" \
  --language_bias_value "${LANGUAGE_BIAS_VALUE}" \
  --language_head_bias_value "${LANGUAGE_HEAD_BIAS_VALUE}" \
  --language_guidance_scope "${LANGUAGE_GUIDANCE_SCOPE}" \
  --use_hydralora_experts "${USE_HYDRALORA_EXPERTS}" \
  --hydralora_num_experts "${HYDRALORA_NUM_EXPERTS}" \
  --hydralora_top_k "${HYDRALORA_TOP_K}" \
  ${ADDITIONAL_TARGET:+--additional_target "${ADDITIONAL_TARGET}"} \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] HydraLoRA LPR job completed"
touch "${OUTPUT_DIR}/.training_complete"
