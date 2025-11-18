#!/bin/bash
set -euo pipefail

ROOTS=(
  "/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_llama31_8b_acc/nopissa"
  "/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_llama31_8b_acc/pissa"
)
LABELS=("simpe_cola_nopissa" "moe_cola_pissa")
TASKS="${TASKS:-belebele,xnli,arc_multilingual,flores}"
WANDB_PROJECT="${WANDB_PROJECT:-lm_eval_project}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

mkdir -p logs

for idx in "${!ROOTS[@]}"; do
  root="${ROOTS[$idx]}"
  label="${LABELS[$idx]}"

  sbatch \
    --job-name="eval-${label}" \
    --output="logs/${label}-%j.out" \
    --export=ALL,CKPT_ROOT="${root}",RUN_LABEL="${label}",TASKS="${TASKS}",WANDB_PROJECT="${WANDB_PROJECT}",LM_EVAL_BIN="${LM_EVAL_BIN}" \
    scripts/eval/all_checkpoints_lm_eval.sh
done
