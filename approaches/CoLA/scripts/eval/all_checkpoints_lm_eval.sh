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
RUN_NAME="${RUN_NAME:-${RUN_LABEL}-$(date +%Y%m%d%H%M%S)}"
TASKS="${TASKS:-belebele,xnli,arc_multilingual,flores}"
WANDB_PROJECT="${WANDB_PROJECT:-lm_eval_project}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

mkdir -p logs

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
WANDB_ARGS_COMMON="project=${WANDB_PROJECT},group=lm_eval,name=${RUN_NAME},id=${RUN_NAME},resume=${WANDB_RESUME}"

for ckpt in "$CKPT_ROOT"/checkpoint-*; do
  [[ -d "$ckpt" ]] || continue

  name=$(basename "$ckpt")
  out="$ckpt/lm_eval"
  mkdir -p "$out"
  wandb="${WANDB_ARGS_COMMON},tags=${name}"

  WANDB_INIT_TIMEOUT="$WANDB_INIT_TIMEOUT" "$LM_EVAL_BIN" \
    --model hf \
    --model_args "pretrained=$ckpt,tokenizer=$ckpt" \
    --tasks "$TASKS" \
    --device cuda \
    --batch_size auto \
    --output_path "$out" \
    --wandb_args "$wandb"
done
