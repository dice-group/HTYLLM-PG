#!/bin/bash
#SBATCH --job-name=fw2_sample
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=normal

set -euo pipefail

SCRIPT_DIR="$(pwd)"
PLAN_INPUT="${1:-/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier3_200langs.csv}"
TOKENIZER_FLAG="${2:-}"
PLAN_CSV="$(realpath "${PLAN_INPUT}")"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_200_tier_samples}"
MAX_SAMPLE="${MAX_SAMPLE:-}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs"

cd "${SCRIPT_DIR}"

CLI_ARGS=( "${PLAN_CSV}" "${OUTPUT_DIR}" )
[[ -n "${MAX_SAMPLE}" ]] && CLI_ARGS+=( "--max-sample" "${MAX_SAMPLE}" )
[[ -n "${TOKENIZER_FLAG}" ]] && CLI_ARGS+=( "--tokenizer-training" )

echo "Sampling plan: ${PLAN_CSV}"
echo "Output directory: ${OUTPUT_DIR}"
[[ -n "${MAX_SAMPLE}" ]] && echo "Max sample per language: ${MAX_SAMPLE}"
[[ -n "${TOKENIZER_FLAG}" ]] && echo "Tokenizer training mode: enabled"

srun python sample_generator_datatrove.py "${CLI_ARGS[@]}"
