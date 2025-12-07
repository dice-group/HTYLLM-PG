#!/bin/bash
#SBATCH --job-name=fw2_sample
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PLAN="${REPO_ROOT}/data_prep/processed_artifacts/sampling_plan_tier3_200langs.csv"
PLAN_INPUT="${1:-${DEFAULT_PLAN}}"
PLAN_CSV="$(realpath "${PLAN_INPUT}")"
MAX_SAMPLE="${MAX_SAMPLE:-10_000_000}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/inital_samples_datatrove}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs"

echo "Sampling plan: ${PLAN_CSV}"
echo "Per-language max sample: ${MAX_SAMPLE}"
echo "Output directory: ${OUTPUT_DIR}"

srun python sample_generator_datatrove.py "${PLAN_CSV}" "${MAX_SAMPLE}" "${OUTPUT_DIR}"
