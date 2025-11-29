#!/bin/bash
#SBATCH --job-name=fw2_sample
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=normal

set -euo pipefail

PLAN_CSV="/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/data_prep/processed_artifacts/filtered_languages.csv"
MAX_SAMPLE="100_000"
OUTPUT_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/inital_samples_datatrove"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs"

srun python sample_generator_datatrove.py "${PLAN_CSV}" "${MAX_SAMPLE}" "${OUTPUT_DIR}"
