#!/bin/bash
#SBATCH --job-name=fw-combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/combine_%j.out
#SBATCH --error=logs/combine_%j.err
#SBATCH --partition=normal

set -euo pipefail

TOKENIZED_DIR="/scratch/hpc-prf-merlin/joel/tokenized_fw"
COMBINED_OUTPUT="/scratch/hpc-prf-merlin/joel/tokenized_fw_combined"
MANIFEST_PATH="/scratch/hpc-prf-merlin/project_data/moe_study/fw_split/split_manifest.json"

mkdir -p "$(dirname "$COMBINED_OUTPUT")" logs

srun python -u combine_tokenized.py \
  --tokenized_dir "$TOKENIZED_DIR" \
  --output_dir "$COMBINED_OUTPUT" \
  --manifest "$MANIFEST_PATH" \
  --overwrite
