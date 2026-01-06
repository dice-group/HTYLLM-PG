#!/bin/bash
set -euo pipefail

REPO_ROOT="/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA"
CKPT="/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260106_173545/checkpoint-80_adapter"
TOKENIZER="meta-llama/Llama-3.1-8B"
OUT="/scratch/hpc-prf-merlin/project_data/moe_study/lm_eval_smoke"
PROJ="htyllm-lm-eval"
GROUP="cola_colaflat_ckpt80_chunks"
CONDA_BASE="/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3"
CONDA_ENV="merlin"

cd "$REPO_ROOT"
mapfile -t BELEBELE < <(grep "^belebele_" configs/lm_eval_tasks.txt)
CHUNK_SIZE=$(( ( ${#BELEBELE[@]} + 4 - 1 ) / 4 ))

for i in 0 1 2 3; do
  TASKS=$(IFS=,; echo "${BELEBELE[@]:$((i*CHUNK_SIZE)):$CHUNK_SIZE}")
  srun --gres=gpu:h100:1 --cpus-per-task=4 --mem=128G --time=06:00:00 --partition=gpu \
    bash -lc "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${CONDA_ENV}'
    cd '${REPO_ROOT}'
    python3 scripts/lm_eval_language_ids.py \
      --checkpoint '${CKPT}' \
      --tokenizer '${TOKENIZER}' \
      --tasks '${TASKS}' \
      --output-dir '${OUT}/belebele_chunk$((i+1))' \
      --batch-size 1 \
      --limit 100 \
      --mode both \
      --wandb-args 'project=${PROJ},group=${GROUP},name=belebele_chunk$((i+1)),mode=online'
    " &
done

srun --gres=gpu:h100:1 --cpus-per-task=4 --mem=128G --time=06:00:00 --partition=gpu \
  bash -lc "
  source '${CONDA_BASE}/etc/profile.d/conda.sh'
  conda activate '${CONDA_ENV}'
  cd '${REPO_ROOT}'
  python3 scripts/lm_eval_language_ids.py \
    --checkpoint '${CKPT}' \
    --tokenizer '${TOKENIZER}' \
    --tasks xnli \
    --output-dir '${OUT}/xnli' \
    --batch-size 1 \
    --limit 100 \
    --mode both \
    --wandb-args 'project=${PROJ},group=${GROUP},name=xnli,mode=online'
  " &
