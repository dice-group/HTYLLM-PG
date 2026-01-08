#!/bin/bash
#SBATCH --job-name=cola-lpr-accelerate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=35:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_acc_moe_cola_lpr_%j.log
#SBATCH --partition=gpu

set -euo pipefail
module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate cola_llama_factory
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"

export WANDB_PROJECT="llama3.1-8b_moe_cola_lpr"
export WANDB_RUN_GROUP="cola_moe_accelerate_lpr"

DATASET_DIR=./LLaMA-Factory/data
DATASET_NAME=c4
FINETUNING_TYPE=cola
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_llama31_8b_acc_lpr}
MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B
ACCEL_CONFIG=./LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml
export ACCELERATE_USE_FSDP=1
TOKENIZED_PATH=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/5_langs

############ Language Prior Routing knobs ############
LANGUAGE_COLUMN=${LANGUAGE_COLUMN:-language}
LANGUAGE_MAP=${LANGUAGE_MAP:-}
LANGUAGE_ROUTER_MODE=${LANGUAGE_ROUTER_MODE:-bias}
LANGUAGE_PRIOR_WEIGHT=${LANGUAGE_PRIOR_WEIGHT:-0.1}
LANGUAGE_BIAS_VALUE=${LANGUAGE_BIAS_VALUE:-5.0}
if [[ -z "${LANGUAGE_MAP}" ]]; then
  echo "[ERROR] LANGUAGE_MAP must point to a JSON file (or inline JSON string) mapping languages to families." >&2
  exit 1
fi
LANGUAGE_ARGS=(
  --language_column "${LANGUAGE_COLUMN}"
  --language_map "${LANGUAGE_MAP}"
  --language_router_mode "${LANGUAGE_ROUTER_MODE}"
  --language_prior_weight "${LANGUAGE_PRIOR_WEIGHT}"
  --language_bias_value "${LANGUAGE_BIAS_VALUE}"
)

############ Training hyperparameters ############
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=5e-5
LR_SCHEDULER_TYPE=cosine
WARMUP_RATIO=0.06
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
NUM_A=${NUM_A:-2}
NUM_B=${NUM_B:-4}
LORA_RANK=4
LORA_ALPHA=8
COLA_NUM_EXPERTS=4
COLA_TOP_K=1
EVAL_STEPS=200
SEED=42
LOGGING_STEPS=10
LOGGING_FIRST_STEP=True
DISABLE_TQDM=False
BF16=True
FP16=False
USE_COLA_EXPERTS=True
USE_REENTRANT_GC=False
USE_COLA_PISSA_INIT=${USE_COLA_PISSA_INIT:-True}
COLA_INIT_LORA_WEIGHTS=${COLA_INIT_LORA_WEIGHTS:-}
if [[ -z "${PURE_BF16:-}" ]]; then
  if [[ "${ACCEL_CONFIG}" == *"fsdp"* ]]; then
    PURE_BF16=True
  else
    PURE_BF16=False
  fi
fi

NUM_LANGS=$(echo "${TOKENIZED_PATH}" | sed -n 's#.*/\([0-9]\+\)_langs.*#\1#p')
NUM_LANGS=${NUM_LANGS:-all}

RUN_NAME="cola_moe_acc_lpr_${NUM_LANGS}langs_a${NUM_A}_b${NUM_B}_$(date +%Y%m%d_%H%M%S)"
WANDB_TAGS="cola,moe,accelerate,bf16,lpr"
export WANDB_TAGS

TOKENIZED_ARGS=()
if [[ -n "${TOKENIZED_PATH}" ]]; then
  if [[ ! -d "${TOKENIZED_PATH}" ]]; then
    echo "[ERROR] TOKENIZED_PATH ${TOKENIZED_PATH} does not exist." >&2
    exit 1
  fi
  TOKENIZED_ARGS+=(--tokenized_path "${TOKENIZED_PATH}")
fi
COLA_INIT_ARGS=()
if [[ -n "${COLA_INIT_LORA_WEIGHTS}" ]]; then
  COLA_INIT_ARGS+=(--cola_init_lora_weights "${COLA_INIT_LORA_WEIGHTS}")
fi

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Starting Accelerate-backed MoE CoLA (LPR) training at $(date)"
accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  ./LLaMA-Factory/src/train.py \
  --stage sft \
  --do_train \
  --do_eval \
  --eval_strategy steps \
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
  --num_A ${NUM_A} \
  --num_B ${NUM_B} \
  --lora_rank ${LORA_RANK} \
  --lora_alpha ${LORA_ALPHA} \
  --use_cola_experts ${USE_COLA_EXPERTS} \
  --cola_num_experts ${COLA_NUM_EXPERTS} \
  --cola_top_k ${COLA_TOP_K} \
  --use_cola_pissa_init ${USE_COLA_PISSA_INIT} \
  "${COLA_INIT_ARGS[@]}" \
  --bf16 $(echo "${BF16}" | tr '[:upper:]' '[:lower:]') \
  --fp16 $(echo "${FP16}" | tr '[:upper:]' '[:lower:]') \
  --pure_bf16 $(echo "${PURE_BF16}" | tr '[:upper:]' '[:lower:]') \
  --disable_tqdm ${DISABLE_TQDM} \
  --logging_steps ${LOGGING_STEPS} \
  --logging_first_step ${LOGGING_FIRST_STEP} \
  --cola_debug \
  --dataloader_num_workers 8 \
  --preprocessing_num_workers 16 \
  --report_to wandb \
  --ddp_find_unused_parameters False \
  "${TOKENIZED_ARGS[@]}" \
  "${LANGUAGE_ARGS[@]}"

echo "[INFO] Training finished at $(date)"
touch "${OUTPUT_DIR}/.training_complete"
echo "[INFO] Wrote completion marker to ${OUTPUT_DIR}/.training_complete"
