#!/bin/bash
#SBATCH --job-name=lm-eval
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lm-eval-%j.out

set -euo pipefail

: "${CKPT_ROOT:?Need CKPT_ROOT pointing to a checkpoint directory root}"
RUN_LABEL="${RUN_LABEL:-$(basename "$CKPT_ROOT")}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-$RUN_LABEL}"
TASKS="${TASKS:-belebele_tel_Telu}"
WANDB_PROJECT="${WANDB_PROJECT:-lm_eval_project}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
WANDB_RESUME="${WANDB_RESUME:-never}"

mkdir -p logs

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
WANDB_ARGS_BASE="project=${WANDB_PROJECT},group=${WANDB_GROUP},resume=${WANDB_RESUME}"

for ckpt in "$CKPT_ROOT"/checkpoint-*; do
  [[ -d "$ckpt" ]] || continue

  name=$(basename "$ckpt")
  out="$ckpt/lm_eval"
  mkdir -p "$out"
  run_name="${RUN_LABEL}-${name}-${RUN_TAG}"
  wandb="${WANDB_ARGS_BASE},name=${run_name},id=${run_name},tags=${name}"

  WANDB_INIT_TIMEOUT="$WANDB_INIT_TIMEOUT" "$LM_EVAL_BIN" \
    --model hf \
    --model_args "pretrained=$ckpt,tokenizer=$ckpt" \
    --tasks "$TASKS" \
    --device cuda \
    --batch_size auto \
    --limit 100 \
    --output_path "$out" \
    --wandb_args "$wandb"
done
