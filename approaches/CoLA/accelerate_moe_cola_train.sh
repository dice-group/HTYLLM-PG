#!/usr/bin/env bash
# -------------------------------------------------
# accelerate_moe_cola_train.sh
# -------------------------------------------------
set -euo pipefail

# ---- user‑supplied hyper‑parameters (with defaults) ----
LR=${LR:-5e-5}
BATCH_SIZE=${BATCH_SIZE:-16}
SEED=${SEED:-42}

#SBATCH --job-name=cola-moe-accelerate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_acc_moe_cola_%j.log
#SBATCH --partition=gpu

module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate cola_llama_factory
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"

export WANDB_RUN_GROUP="grid_lr${LR}_bs${BATCH_SIZE}"
export WANDB_PROJECT="llama3.1-8b_moe_cola_training_accelerate"
export WANDB_RUN_GROUP="cola_moe_accelerate"

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_llama31_8b_acc
MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B
ACCEL_CONFIG=./LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml
TOKENIZED_PATH=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/5_langs

NUM_LANGS=$(echo "${TOKENIZED_PATH}" | sed -n 's#.*/\([0-9]\+\)_langs.*#\1#p')
NUM_LANGS=${NUM_LANGS:-all}

RUN_NAME="cola_moe_acc_${NUM_LANGS}langs_$(date +%Y%m%d_%H%M%S)"
WANDB_TAGS="cola,moe,accelerate,bf16"
WANDB_CONFIG_JSON=$(cat <<JSON
{
  "model_name_or_path": "${MODEL_NAME_OR_PATH}",
  "tokenized_path": "${TOKENIZED_PATH}",
  "finetuning_type": "cola",
  "dataset": "c4",
  "num_A": 1,
  "num_B": 1,
  "lora_rank": 4,
  "lora_alpha": 8,
  "cola_num_experts": 2,
  "cola_top_k": 2,
  "per_device_train_batch_size": 1,
  "gradient_accumulation": 8,
  "eval_fraction_per_language": 0.05,
  "num_langs": "${NUM_LANGS}"
}
JSON
)
export WANDB_TAGS
export WANDB_CONFIG_JSON

LM_EVAL_TASKS=belebele
LM_EVAL_BATCH_SIZE=auto
LM_EVAL_OUTPUT_DIR=${OUTPUT_DIR}/lm_eval
LM_EVAL_WANDB_PROJECT=llama31_multilingual_eval_belebele
LM_EVAL_WANDB_PREFIX=cola_moe_acc
LM_EVAL_VISIBLE_GPUS=
LM_EVAL_EXTRA_ARGS=

if [[ ! -f "$ACCEL_CONFIG" ]]; then
  echo "[ERROR] Accelerate config $ACCEL_CONFIG not found." >&2
  exit 1
fi

TOKENIZED_ARGS=()
if [[ -n "$TOKENIZED_PATH" ]]; then
  if [[ ! -d "$TOKENIZED_PATH" ]]; then
    echo "[ERROR] TOKENIZED_PATH ${TOKENIZED_PATH} does not exist." >&2
    exit 1
  fi
  TOKENIZED_ARGS+=(--tokenized_path "${TOKENIZED_PATH}")
fi

export ACCELERATE_USE_FSDP=1

echo "[INFO] Starting Accelerate-backed MoE CoLA training at $(date)"
srun accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  ./LLaMA-Factory/src/train.py \
  --stage sft \
  --do_train \
  --run_name "${RUN_NAME}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset c4 \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type cola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --learning_rate $LR \
  --num_train_epochs 1 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size 1 \
  --seed $SEED \
  --num_A 2 \
  --num_B 4 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --use_cola_experts \
  --cola_num_experts 4 \
  --cola_top_k 2 \
  --bf16 True \
  --fp16 False \
  --cola_debug \
  --report_to wandb \
  "${TOKENIZED_ARGS[@]}"

echo "[INFO] Training finished at $(date)"
  #--do_eval \
  #--evaluation_strategy steps \
  #--eval_steps 500  \
echo "[INFO] Collecting checkpoints for evaluation..."
mapfile -t CHECKPOINTS < <(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  CHECKPOINTS=("${OUTPUT_DIR}")
fi

if [[ -n "$LM_EVAL_VISIBLE_GPUS" ]]; then
  CUDA_PREFIX=(CUDA_VISIBLE_DEVICES="$LM_EVAL_VISIBLE_GPUS")
else
  CUDA_PREFIX=()
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  ckpt_name=$(basename "$checkpoint")
  ckpt_label=${ckpt_name:-final}
  out_file="${LM_EVAL_OUTPUT_DIR}/${ckpt_label}_lm_eval.jsonl"
  wandb_name="${LM_EVAL_WANDB_PREFIX}_${ckpt_label}"

  echo "[INFO] Running lm-eval on ${ckpt_label}, writing to ${out_file}"
  "${CUDA_PREFIX[@]}" lm_eval \
    --model hf \
    --model_args "pretrained=${checkpoint},tokenizer=${checkpoint}" \
    --tasks "${LM_EVAL_TASKS}" \
    --batch_size "${LM_EVAL_BATCH_SIZE}" \
    --output_path "${out_file}" \
    --wandb_args "project=${LM_EVAL_WANDB_PROJECT},name=${wandb_name}" \
    ${LM_EVAL_EXTRA_ARGS}
done

echo "[INFO] Accelerate workflow finished at $(date)"
