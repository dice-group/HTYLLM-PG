#!/bin/bash
#SBATCH --job-name=fw2_sample
#SBATCH --partition=normal
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=logs/sample_%j.out

PLAN_CSV="/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/data_prep/processed_artifacts/filtered_languages.csv"
MAX_SAMPLE="100_000"
OUTPUT_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset"

srun python sample_generator_datatrove.py "${PLAN_CSV}" "${MAX_SAMPLE}" "${OUTPUT_DIR}"
