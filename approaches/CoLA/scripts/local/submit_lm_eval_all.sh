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
LAST_JOB_ID=""

for CKPT in "${CKPTS[@]}"; do
  RUN_NAME="$(basename "$(dirname "${CKPT}")")"
  CKPT_NAME="$(basename "${CKPT}")"
  OUT_DIR="${OUT}/all/${RUN_NAME}/${CKPT_NAME}"
  DEP_OPT=()
  if [[ -n "${LAST_JOB_ID}" ]]; then
    DEP_OPT=(--dependency="afterok:${LAST_JOB_ID}")
  fi
  JOB_SUBMIT_OUT=$(sbatch "${DEP_OPT[@]}" \
    --job-name="lm-eval-${RUN_NAME}" --output="${LOG_DIR}/lm-eval_${RUN_NAME}_%j.log" \
    --gres=gpu:h100:1 --cpus-per-task=4 --mem=128G --time=12:00:00 --partition=gpu \
    --export=ALL,CKPT="${CKPT}",RUN_NAME="${RUN_NAME}" \
    --wrap "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${CONDA_ENV}'
    cd '${REPO_ROOT}'
    python3 scripts/lm_eval_language_ids.py \
      --checkpoint \"${CKPT}\" \
      --tokenizer '${TOKENIZER}' \
      --tasks '${TASKS}' \
      --output-dir \"${OUT_DIR}\" \
      --batch-size auto \
      --device-map auto \
      --limit 10 \
      --mode both \
      --wandb-args \"project=${PROJ},group=${GROUP},name=${RUN_NAME},mode=online\"
    echo '========== DETAILED EVAL FINISHED =========='
    echo '========== BEGIN SUMMARY STEP =========='
    sleep 10
    python3 scripts/wandb_summary_job.py \
      --checkpoint \"${CKPT}\" \
      --output-dir \"${OUT_DIR}\" \
      --wandb-args \"project=${PROJ},group=${GROUP},name=${RUN_NAME},mode=online\"
    "
  )
  LAST_JOB_ID=$(echo "${JOB_SUBMIT_OUT}" | awk '{print $4}')
done
