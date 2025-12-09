#!/bin/bash
#SBATCH --job-name=tokenizer_extension
#SBATCH --output=logs/tokenize_extension/tokenizer_extension_%j.out
#SBATCH --error=logs/tokenize_extension/tokenizer_extension_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=420G
#SBATCH -p normal

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
LOG_DIR="${SCRIPT_DIR}/logs/tokenize_extension"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CONFIG_PATH="${1:-${TOKENIZER_EXTENSION_CONFIG:-${SCRIPT_DIR}/configs/cola_tier1_12langs.yaml}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[run_pipeline_slurm] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[run_pipeline_slurm] Launching tokenizer extension pipeline with config ${CONFIG_PATH}"
srun bash -c "cd '${PROJECT_ROOT}' && PYTHONPATH='${PROJECT_ROOT}' python -m tokenizer_extension.pipeline --config '${CONFIG_PATH}'"
