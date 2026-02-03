#!/usr/bin/env bash
# Evaluates the hard-coded Llama-3.1-8B model on every Belebele subset, logs to W&B, and saves JSON/CSV summaries.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
CKPT="meta-llama/Llama-3.1-8B"
TOKENIZER="meta-llama/Llama-3.1-8B"
OUTDIR="data_prep/processed_artifacts/lm_eval/llama31_8b"

# Discover Belebele languages, otherweise it uses weird varaints such as belebele_amh_prompt_2 or belebele_afr_prompt_5
BELEBELE_TASKS=$(ls lm_eval/tasks/belebele \
  | grep '^belebele_[a-z][a-z][a-z]_' \
  | grep -v 'prompt' \
  | sed 's/.yaml//' \
  | sort -u \
  | paste -sd ',' -)

TASKS="${BELEBELE_TASKS}"

WANDB_PROJECT=${WANDB_PROJECT:-"llama3.1_8b_multilingual_eval"}
WANDB_NAME=${WANDB_NAME:-"lm_eval_belebele_llama31_8b"}
BS=${BS:-"auto"}

mkdir -p "$OUTDIR"
JSON_PATH="${OUTDIR}/lm_eval.json"
CSV_PATH="${OUTDIR}/lm_eval.csv"

echo "Running lm-eval on Belebele tasks: ${TASKS}"

lm_eval \
  --model hf \
  --model_args "pretrained=${CKPT},tokenizer=${TOKENIZER}" \
  --tasks "${TASKS}" \
  --batch_size "${BS}" \
  --limit 250 \
  --output_path "${JSON_PATH}" \
  --wandb_args "project=${WANDB_PROJECT},name=${WANDB_NAME}"