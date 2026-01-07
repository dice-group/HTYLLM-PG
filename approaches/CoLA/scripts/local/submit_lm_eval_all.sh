#!/bin/bash
set -euo pipefail

REPO_ROOT="/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA"
CKPTS=(
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260106_173545/checkpoint-40_adapter"
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260106_173545/checkpoint-80_adapter"
)
TOKENIZER="meta-llama/Llama-3.1-8B"
OUT="/scratch/hpc-prf-merlin/project_data/moe_study/lm_eval_smoke"
PROJ="htyllm-lm-eval"
GROUP="cola_colaflat_ckpt80_all"
CONDA_BASE="/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3"
CONDA_ENV="merlin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

cd "$REPO_ROOT"
mkdir -p "${LOG_DIR}"
TASKS="belebele_eng_Latn,belebele_deu_Latn,belebele_zul_Latn"

for CKPT in "${CKPTS[@]}"; do
  RUN_NAME="$(basename "$(dirname "${CKPT}")")"
  EVAL_JOB=$(sbatch --parsable --job-name="lm-eval-${RUN_NAME}" --output="${LOG_DIR}/lm-eval_${RUN_NAME}_%j.log" \
    --gres=gpu:h100:1 --cpus-per-task=4 --mem=400G --time=12:00:00 --partition=gpu \
    --export=ALL,CKPT="${CKPT}",RUN_NAME="${RUN_NAME}" \
    --wrap "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${CONDA_ENV}'
    cd '${REPO_ROOT}'
    python3 scripts/lm_eval_language_ids.py \
      --checkpoint \"${CKPT}\" \
      --tokenizer '${TOKENIZER}' \
      --tasks '${TASKS}' \
      --output-dir '${OUT}/all' \
      --batch-size auto \
      --device-map auto \
      --limit 500 \
      --mode both \
      --wandb-args \"project=${PROJ},group=${GROUP},name=${RUN_NAME},mode=online\"
    ")

  sbatch --job-name="lm-eval-summary-${RUN_NAME}" --output="${LOG_DIR}/lm-eval_summary_${RUN_NAME}_%j.log" \
    --dependency=afterok:${EVAL_JOB} \
    --cpus-per-task=1 --mem=8G --time=00:30:00 --partition=normal \
    --export=ALL,CKPT=\"${CKPT}\",RUN_NAME=\"${RUN_NAME}\" \
    --wrap "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${CONDA_ENV}'
    cd '${REPO_ROOT}'
    python3 scripts/wandb_summary_job.py \
      --checkpoint \"${CKPT}\" \
      --output-dir \"${OUT}/all\" \
      --wandb-args \"project=${PROJ},group=${GROUP},name=${RUN_NAME},mode=online\"
    "
done
