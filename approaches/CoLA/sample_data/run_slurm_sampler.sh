#!/bin/bash
#SBATCH --job-name=fw2_sample
#SBATCH --partition=normal
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <plan_csv> <max_sample> <output_dir>"
    exit 1
fi

PLAN_CSV="$1"
MAX_SAMPLE="$2"
OUTPUT_DIR="$3"

srun python sample_data/sample_generator_datatrove.py "${PLAN_CSV}" "${MAX_SAMPLE}" "${OUTPUT_DIR}"
