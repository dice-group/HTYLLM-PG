#!/bin/bash
#SBATCH --job-name=lpr-cola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
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
export WANDB_NAME="${WANDB_NAME:-cola-lpr}"
export WANDB_TAGS="${WANDB_TAGS:-comparison,cola,lpr}"
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
DIST_DEBUG="${DIST_DEBUG:-1}"
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
USE_COLA_EXPERTS="${USE_COLA_EXPERTS:-True}"
BF16="${BF16:-True}"
FP16="${FP16:-False}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-false}"

TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_LR="${LEARNING_RATE:-2e-4}"
TRAIN_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
EVAL_BS="${PER_DEVICE_EVAL_BATCH_SIZE:-${TRAIN_BS}}"
GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_EXPERTS="${COLA_NUM_EXPERTS:-4}"
TOP_K="${COLA_TOP_K:-2}"
NUM_A="${COLA_NUM_A:-2}"
NUM_B="${COLA_NUM_B:-3}"
EXPERT_NUM_B="${COLA_EXPERT_NUM_B:-}"
ADDITIONAL_TARGET="${ADDITIONAL_TARGET:-}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
STRATEGY="${COLA_STRATEGY:-fully}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"

RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))
RUN_NAME="${WANDB_NAME}_ep${TRAIN_EPOCHS}_bs${TRAIN_BS}x${GRAD_ACC}_lr${TRAIN_LR}_nexp${NUM_EXPERTS}_k${TOP_K}_${STRATEGY}_${RANDOM_ID}"
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
  "cola_strategy": "${STRATEGY}",
  "use_cola_experts": ${USE_COLA_EXPERTS},
  "cola_num_experts": ${NUM_EXPERTS},
  "cola_top_k": ${TOP_K}
}
EOF
)

echo "[INFO] Running CoLA LPR training into ${OUTPUT_DIR}"

python "${REPO_ROOT}/scripts/comparison/router_setup.py" --type cola

ACCELERATE_CMD=()
ENTRYPOINT=()
LAUNCH_PREFIX=()
if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  # Deterministic rendezvous (avoid probe-then-use races with torch elastic).
  MASTER_HOST=""
  if [[ -n "${MASTER_ADDR_OVERRIDE:-}" ]]; then
    MASTER_ADDR="${MASTER_ADDR_OVERRIDE}"
    MASTER_HOST="${MASTER_ADDR_OVERRIDE}"
  elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    # Always use the same master host across nodes (avoid per-node hostname).
    MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_ADDR=$(getent ahosts "${MASTER_HOST}" | awk '$1 ~ /^[0-9]+(\\.[0-9]+){3}$/ {print $1; exit}')
    if [[ -z "${MASTER_ADDR}" ]]; then
      MASTER_ADDR="${MASTER_HOST}"
    fi
  else
    MASTER_ADDR="127.0.0.1"
    MASTER_HOST="${MASTER_ADDR}"
  fi
  MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000) ))}"
  export MASTER_ADDR MASTER_PORT
  echo "[INFO] accelerate rendezvous MASTER_HOST=${MASTER_HOST} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
  if [[ "${DIST_DEBUG:-}" == "1" && "${SLURM_NNODES:-1}" -gt 1 ]]; then
    srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --export=ALL \
      bash -c "echo host=\$(hostname) nodeid=\${SLURM_NODEID} master_host=${MASTER_HOST} master_addr=${MASTER_ADDR}; getent ahosts ${MASTER_HOST} | head -n 3"
  fi
  if [[ "${SLURM_NNODES:-1}" -gt 1 && "${SLURM_NODEID:-0}" -ne 0 ]]; then
    bash "${REPO_ROOT}/scripts/comparison/wait_for_master.sh" \
      "${MASTER_ADDR}" "${MASTER_PORT}" "${MASTER_CONNECT_TIMEOUT:-90}"
  fi
  ACCELERATE_CMD=(
    accelerate launch
    --config_file "${ACCELERATE_CONFIG_FILE}"
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT}"
  )
  if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    GPUS_PER_NODE_SOURCE="default"
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      GPUS_PER_NODE_SOURCE="SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
      GPUS_PER_NODE=$(echo "${SLURM_GPUS_ON_NODE}" | awk -F: '{print $NF}' | grep -oE '[0-9]+' | head -n 1)
    elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
      GPUS_PER_NODE_SOURCE="SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
      IFS=',' read -ra _gpu_list <<< "${SLURM_JOB_GPUS}"
      GPUS_PER_NODE=${#_gpu_list[@]}
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      GPUS_PER_NODE_SOURCE="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
      GPUS_PER_NODE=${#_gpu_list[@]}
    else
      GPUS_PER_NODE=4
    fi
    TOTAL_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))
    echo "[INFO] dist config: nnodes=${SLURM_NNODES} node_id=${SLURM_NODEID:-0} gpus_per_node=${GPUS_PER_NODE} total_procs=${TOTAL_PROCESSES} source=${GPUS_PER_NODE_SOURCE}"
    ACCELERATE_CMD+=(--num_machines "${SLURM_NNODES}" --machine_rank "${SLURM_NODEID:-0}" --num_processes "${TOTAL_PROCESSES}")
    LAUNCH_PREFIX=(srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --export=ALL)
  fi
  ACCELERATE_CMD+=(--module)
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

DEBUG_FLAGS=()
if [[ "${COLA_DEBUG:-}" == "true" || "${COLA_DEBUG:-}" == "True" || "${COLA_DEBUG:-}" == "1" ]]; then
  DEBUG_FLAGS+=(--cola_debug)
fi

"${LAUNCH_PREFIX[@]}" "${ACCELERATE_CMD[@]}" "${ENTRYPOINT[@]}" \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET_NAME}" \
  --eval_dataset "${EVAL_DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type cola \
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
  --eval_steps "${EVAL_STEPS:-5000}" \
  --save_steps "${SAVE_STEPS:-5000}" \
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
  --num_A "${NUM_A}" \
  --num_B "${NUM_B}" \
  ${EXPERT_NUM_B:+--cola_expert_num_B "${EXPERT_NUM_B}"} \
  --cola_strategy "${STRATEGY}" \
  --use_cola_experts "${USE_COLA_EXPERTS}" \
  --cola_num_experts "${NUM_EXPERTS}" \
  --cola_top_k "${TOP_K}" \
  --language_column "${LANGUAGE_COLUMN}" \
  --language_map "${LANGUAGE_MAP}" \
  --language_router_mode "${LANGUAGE_ROUTER_MODE}" \
  --language_head_router_mode "${LANGUAGE_HEAD_ROUTER_MODE}" \
  --language_prior_weight "${LANGUAGE_PRIOR_WEIGHT}" \
  --language_bias_value "${LANGUAGE_BIAS_VALUE}" \
  --language_head_bias_value "${LANGUAGE_HEAD_BIAS_VALUE}" \
  --language_guidance_scope "${LANGUAGE_GUIDANCE_SCOPE}" \
  "${DEBUG_FLAGS[@]}" \
  ${ADDITIONAL_TARGET:+--additional_target "${ADDITIONAL_TARGET}"} \
  --report_to wandb \
  --include_effective_tokens_per_second true \
  --include_num_input_tokens_seen true

echo "[INFO] CoLA LPR job completed"
touch "${OUTPUT_DIR}/.training_complete"
