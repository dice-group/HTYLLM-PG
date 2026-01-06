#!/bin/bash
set -euo pipefail

REPO_ROOT="/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA"
CKPT="/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260106_173545/checkpoint-80_adapter"
TOKENIZER="meta-llama/Llama-3.1-8B"
OUT="/scratch/hpc-prf-merlin/project_data/moe_study/lm_eval_smoke"
PROJ="htyllm-lm-eval"
GROUP="cola_colaflat_ckpt80_all"
CONDA_BASE="/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3"
CONDA_ENV="merlin"
LOG_DIR="$(pwd)/logs"

cd "$REPO_ROOT"
mkdir -p "${LOG_DIR}"
TASKS=$(paste -sd, configs/lm_eval_tasks.txt | tr -d '\r')

sbatch --job-name="lm-eval-all" --output="${LOG_DIR}/lm-eval_all_%j.log" \
  --gres=gpu:h100:1 --cpus-per-task=4 --mem=400G --time=12:00:00 --partition=gpu \
  --wrap "
  source '${CONDA_BASE}/etc/profile.d/conda.sh'
  conda activate '${CONDA_ENV}'
  cd '${REPO_ROOT}'
  python3 scripts/lm_eval_language_ids.py \
    --checkpoint '${CKPT}' \
    --tokenizer '${TOKENIZER}' \
    --tasks '${TASKS}' \
    --output-dir '${OUT}/all' \
    --batch-size auto \
    --device-map auto \
    --mode both \
    --wandb-args 'project=${PROJ},group=${GROUP},name=all,mode=online'
  "
