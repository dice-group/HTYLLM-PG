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
LOG_DIR="$(pwd)/logs"

cd "$REPO_ROOT"
mkdir -p "${LOG_DIR}"
mapfile -t BELEBELE < <(grep "^belebele_" configs/lm_eval_tasks.txt | tr -d '\r')
CHUNK_SIZE=$(( ( ${#BELEBELE[@]} + 4 - 1 ) / 4 ))

for i in 0 1 2 3; do
  SLICE=("${BELEBELE[@]:$((i*CHUNK_SIZE)):$CHUNK_SIZE}")
  TASKS=$(IFS=,; echo "${SLICE[*]}")
  sbatch --job-name="lm-eval-bele${i}" --output="${LOG_DIR}/lm-eval_bele${i}_%j.log" \
    --gres=gpu:h100:1 --cpus-per-task=4 --mem=128G --time=06:00:00 --partition=gpu \
    --wrap "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${CONDA_ENV}'
    cd '${REPO_ROOT}'
    python3 scripts/lm_eval_language_ids.py \
      --checkpoint '${CKPT}' \
      --tokenizer '${TOKENIZER}' \
      --tasks '${TASKS}' \
      --output-dir '${OUT}/belebele_chunk$((i+1))' \
      --batch-size 2 \
      --limit 100 \
      --device-map auto \
      --mode both \
      --wandb-args 'project=${PROJ},group=${GROUP},name=belebele_chunk$((i+1)),mode=online'
    "
done

sbatch --job-name="lm-eval-xnli" --output="${LOG_DIR}/lm-eval_xnli_%j.log" \
  --gres=gpu:h100:1 --cpus-per-task=4 --mem=128G --time=06:00:00 --partition=gpu \
  --wrap "
  source '${CONDA_BASE}/etc/profile.d/conda.sh'
  conda activate '${CONDA_ENV}'
  cd '${REPO_ROOT}'
  python3 scripts/lm_eval_language_ids.py \
    --checkpoint '${CKPT}' \
    --tokenizer '${TOKENIZER}' \
    --tasks xnli \
    --output-dir '${OUT}/xnli' \
    --batch-size 2 \
    --limit 100 \
    --device-map auto \
    --mode both \
    --wandb-args 'project=${PROJ},group=${GROUP},name=xnli,mode=online'
  "
