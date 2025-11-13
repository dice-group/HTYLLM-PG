#!/bin/bash
#SBATCH --job-name=cola-moe-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_moe_cola_%j.log
#SBATCH --partition=gpu

set -euo pipefail
module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate cola_llama_factory
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"


export WANDB_PROJECT="llama3.1-8b_moe_cola_training"

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_llama31_8b
MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B
TRAIN_LOG=logs/train_moe_cola_${SLURM_JOB_ID:-manual}.log

LM_EVAL_TASKS=belebele
LM_EVAL_BATCH_SIZE=auto
LM_EVAL_OUTPUT_DIR=${OUTPUT_DIR}/lm_eval
LM_EVAL_WANDB_PROJECT=llama31_multilingual_eval_belebele
LM_EVAL_WANDB_PREFIX=cola_moe
LM_EVAL_VISIBLE_GPUS=
LM_EVAL_EXTRA_ARGS=
TOKENIZED_PATH=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/46_langs

mkdir -p "${OUTPUT_DIR}" "$(dirname "$TRAIN_LOG")" "$LM_EVAL_OUTPUT_DIR"

export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

TOKENIZED_ARGS=()
if [[ -n "$TOKENIZED_PATH" ]]; then
  if [[ ! -d "$TOKENIZED_PATH" ]]; then
    echo "[ERROR] TOKENIZED_PATH ${TOKENIZED_PATH} does not exist." >&2
    exit 1
  fi
  TOKENIZED_ARGS+=(--tokenized_path "${TOKENIZED_PATH}")
fi

echo "[INFO] Starting MoE CoLA training at $(date)"
srun llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --resize_vocab true \
  --dataset c4 \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type cola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_A 1 \
  --num_B 1 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --use_cola_experts \
  --cola_num_experts 2 \
  --cola_top_k 2 \
  --bf16 True \
  --fp16 False \
  --cola_debug \
  "${TOKENIZED_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"

echo "[INFO] Training finished at $(date)"

echo "[INFO] Collecting checkpoints for evaluation..."
mapfile -t CHECKPOINTS < <(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

# If no numbered checkpoints were produced, evaluate the final output directory.
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

echo "[INFO] Evaluation finished at $(date)"
