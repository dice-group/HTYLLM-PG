#!/bin/bash
#SBATCH --job-name=tokenizer_extension
#SBATCH --output=logs/tokenize_extension/tokenizer_extension_%j.log
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=420G
#SBATCH -p normal

set -euo pipefail
mkdir -p "logs/tokenize_extension"
CONFIG_PATH="${1:-${TOKENIZER_EXTENSION_CONFIG:-configs/cola_tier1_12langs.yaml}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[run_pipeline_slurm] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[run_pipeline_slurm] Launching tokenizer extension pipeline with config ${CONFIG_PATH}"
srun python pipeline.py --config "${CONFIG_PATH}"
