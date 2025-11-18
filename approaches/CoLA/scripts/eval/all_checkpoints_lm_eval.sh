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
TASKS="${TASKS:-belebele,xnli,arc_multilingual,flores}"
WANDB_PROJECT="${WANDB_PROJECT:-lm_eval_project}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

mkdir -p logs

for ckpt in "$CKPT_ROOT"/checkpoint-*; do
  [[ -d "$ckpt" ]] || continue

  name=$(basename "$ckpt")
  run_name="${RUN_LABEL}-${name}"
  out="$ckpt/lm_eval"
  mkdir -p "$out"
  wandb="project=${WANDB_PROJECT},group=lm_eval,name=${run_name},id=${run_name}-$(date +%s)"

  "$LM_EVAL_BIN" \
    --model hf \
    --model_args "pretrained=$ckpt,tokenizer=$ckpt" \
    --tasks "$TASKS" \
    --device cuda \
    --output_path "$out" \
    --wandb_args "$wandb"
done
