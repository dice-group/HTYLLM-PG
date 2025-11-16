#!/bin/bash
#SBATCH --job-name=hydralora-moe-train
#SBATCH --nodes=1                    
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_moe_hydralora_%j.log
#SBATCH --partition=gpu

set -euo pipefail

module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

# Use shared scratch HF cache
export HF_HOME=/scratch/hpc-prf-merlin/shared_cache/huggingface/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate hydralora_llama_factory
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
#export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"

export WANDB_PROJECT="llama3.1-8b_moe_hydralora_training_accelerate"
export WANDB_RUN_GROUP="hydralora_moe_accelerate"

DATASET_DIR=./LLaMA-Factory/data
DATASET_NAME=c4
FINETUNING_TYPE=hydralora
OUTPUT_DIR=/scratch/hpc-prf-merlin/sashreek/moe_study/saves/hydralora_moe_llama31_8b_acc
MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B
ACCEL_CONFIG=./LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml
export ACCELERATE_USE_FSDP=1
#TRAIN_LOG=logs/train_moe_hydralora_${SLURM_JOB_ID:-manual}.log
TOKENIZED_PATH=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/46_langs

LM_EVAL_TASKS=${LM_EVAL_TASKS:-belebele}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-auto}
LM_EVAL_WANDB_PROJECT=${LM_EVAL_WANDB_PROJECT:-llama31_multilingual_eval_belebele}
LM_EVAL_WANDB_PREFIX=${LM_EVAL_WANDB_PREFIX:-hydralora_moe_acc}
LM_EVAL_EXTRA_ARGS=${LM_EVAL_EXTRA_ARGS:-}
LM_EVAL_POLL_INTERVAL=${LM_EVAL_POLL_INTERVAL:-120}
ENABLE_LM_EVAL_LISTENER=${ENABLE_LM_EVAL_LISTENER:-1}
CHECKPOINT_LISTENER_SCRIPT=./checkpoint_listener.sh
LM_EVAL_SCRIPT=./lm_eval_checkpoint.sh

############ Training hyperparameters ############
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=5e-5
LR_SCHEDULER_TYPE=cosine
WARMUP_RATIO=0.06
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
LORA_NUM=4
LORA_RANK=4
LORA_ALPHA=8
HYDRALORA_NUM_EXPERTS=4
HYDRALORA_TOP_K=1
EVAL_STEPS=200
SEED=42
LOGGING_STEPS=10
LOGGING_FIRST_STEP=True
DISABLE_TQDM=False
BF16=True
FP16=False
USE_HYDRALORA_EXPERTS=True
USE_REENTRANT_GC=False
if [[ -z "${PURE_BF16:-}" ]]; then
  if [[ "${ACCEL_CONFIG}" == *"fsdp"* ]]; then
    PURE_BF16=True
  else
    PURE_BF16=False
  fi
fi

NUM_LANGS=$(echo "${TOKENIZED_PATH}" | sed -n 's#.*/\([0-9]\+\)_langs.*#\1#p')
NUM_LANGS=${NUM_LANGS:-all}

RUN_NAME="hydralora_moe_acc_${NUM_LANGS}langs_$(date +%Y%m%d_%H%M%S)"
WANDB_TAGS="hydralora,moe,accelerate,bf16"
WANDB_CONFIG_JSON=$(cat <<JSON
{
  "model_name_or_path": "${MODEL_NAME_OR_PATH}",
  "tokenized_path": "${TOKENIZED_PATH}",
  "finetuning_type": "${FINETUNING_TYPE}",
  "dataset": "${DATASET_NAME}",
  "eval_dataset": "${DATASET_NAME}",
  "lora_num": ${LORA_NUM},
  "lora_rank": ${LORA_RANK},
  "lora_alpha": ${LORA_ALPHA},
  "hydralora_num_experts": ${HYDRALORA_NUM_EXPERTS},
  "hydralora_top_k": ${HYDRALORA_TOP_K},
  "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BATCH_SIZE},
  "per_device_eval_batch_size": ${PER_DEVICE_EVAL_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "learning_rate": ${LEARNING_RATE},
  "lr_scheduler_type": "${LR_SCHEDULER_TYPE}",
  "warmup_ratio": ${WARMUP_RATIO},
  "num_train_epochs": ${NUM_TRAIN_EPOCHS},
  "use_hydralora_experts": $(echo "${USE_HYDRALORA_EXPERTS}" | tr '[:upper:]' '[:lower:]'),
  "bf16": $(echo "${BF16}" | tr '[:upper:]' '[:lower:]'),
  "fp16": $(echo "${FP16}" | tr '[:upper:]' '[:lower:]'),
  "seed": ${SEED},
  "logging_steps": ${LOGGING_STEPS},
  "pure_bf16": $(echo "${PURE_BF16}" | tr '[:upper:]' '[:lower:]'),
  "num_langs": "${NUM_LANGS}"
}
JSON
)
export WANDB_TAGS
export WANDB_CONFIG_JSON

if [[ ! -f "$ACCEL_CONFIG" ]]; then
  echo "[ERROR] Accelerate config $ACCEL_CONFIG not found." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

LISTENER_PID=""
# Temporarily disable checkpoint listener during debugging.
# if [[ "${ENABLE_LM_EVAL_LISTENER}" != "0" ]]; then
#   if [[ -x "${CHECKPOINT_LISTENER_SCRIPT}" && -x "${LM_EVAL_SCRIPT}" ]]; then
#     bash "${CHECKPOINT_LISTENER_SCRIPT}" \
#       --watch-dir "${OUTPUT_DIR}" \
#       --eval-script "${LM_EVAL_SCRIPT}" \
#       --tasks "${LM_EVAL_TASKS}" \
#       --batch-size "${LM_EVAL_BATCH_SIZE}" \
#       --poll-interval "${LM_EVAL_POLL_INTERVAL}" \
#       --wandb-project "${LM_EVAL_WANDB_PROJECT}" \
#       --wandb-prefix "${LM_EVAL_WANDB_PREFIX}" \
#       --extra-args "${LM_EVAL_EXTRA_ARGS}" &
#     LISTENER_PID=$!
#     echo "[INFO] Started checkpoint listener (PID ${LISTENER_PID})"
#   else
#     echo "[WARN] Listener or eval script missing/executable bit not set; skipping checkpoint evaluation listener."
#   fi
# else
#   echo "[INFO] LM evaluation listener disabled."
# fi

# # -------- Slurm → torchrun distributed setup --------

# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# MASTER_PORT=${MASTER_PORT:-6000}

# NNODES=${SLURM_NNODES}
# NODE_RANK=${SLURM_NODEID}
# GPUS_PER_NODE=1       # 1 GPU per node (matches --gres)

# export MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE

# # -------- Optional tokenized path handling --------

TOKENIZED_ARGS=()
if [[ -n "$TOKENIZED_PATH" ]]; then
  if [[ ! -d "$TOKENIZED_PATH" ]]; then
    echo "[ERROR] TOKENIZED_PATH ${TOKENIZED_PATH} does not exist." >&2
    exit 1
  fi
  TOKENIZED_ARGS+=(--tokenized_path "${TOKENIZED_PATH}")
fi

# which llamafactory-cli
# python -c "import llamafactory, inspect, sys; print(llamafactory.__file__)"

# echo "[INFO] Starting multi-node HydraLoRA training at $(date)"
# echo "[INFO] MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, NNODES=${NNODES}, NODE_RANK=${NODE_RANK}, GPUS_PER_NODE=${GPUS_PER_NODE}"

# -------- Training with torchrun across nodes --------

echo "[INFO] Starting Accelerate-backed MoE HydraLora training at $(date)"
accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  ./LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --run_name "${RUN_NAME}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset "${DATASET_NAME}" \
    --eval_dataset "${DATASET_NAME}" \
    --dataset_dir "${DATASET_DIR}" \
    --template llama3 \
    --finetuning_type "${FINETUNING_TYPE}" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --seed ${SEED} \
    --lora_num ${LORA_NUM} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --use_hydralora_experts ${USE_HYDRALORA_EXPERTS} \
    --hydralora_num_experts ${HYDRALORA_NUM_EXPERTS} \
    --hydralora_top_k ${HYDRALORA_TOP_K} \
    --bf16 $(echo "${BF16}" | tr '[:upper:]' '[:lower:]') \
    --fp16 $(echo "${FP16}" | tr '[:upper:]' '[:lower:]') \
    --pure_bf16 $(echo "${PURE_BF16}" | tr '[:upper:]' '[:lower:]') \
    --disable_tqdm ${DISABLE_TQDM} \
    --logging_steps ${LOGGING_STEPS} \
    --logging_first_step ${LOGGING_FIRST_STEP} \
    --hydralora_debug \
    --dataloader_num_workers 8 \
    --preprocess_num_workers 16 \
    --report_to wandb \
    --ddp_find_unused_parameters False \
    "${TOKENIZED_ARGS[@]}"

echo "[INFO] Training finished at $(date)"

# -------- Evaluation loop (same as before) --------

echo "[INFO] Collecting checkpoints for evaluation..."
touch "${OUTPUT_DIR}/.training_complete"
echo "[INFO] Wrote completion marker to ${OUTPUT_DIR}/.training_complete"
if [[ -n "$LISTENER_PID" ]]; then
  echo "[INFO] Waiting for checkpoint listener (PID ${LISTENER_PID}) to finish scheduling evaluations..."
  wait "$LISTENER_PID"
fi