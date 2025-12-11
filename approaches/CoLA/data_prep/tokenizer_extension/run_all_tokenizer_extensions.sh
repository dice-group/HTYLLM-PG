#!/usr/bin/env bash
set -euo pipefail

# Submits tokenizer-extension jobs for all CoLA tiers (12, 72, 200 languages).
# Assumes this script is run from data_prep/tokenizer_extension.

LOG_ROOT="logs/tokenize_extension"
SLURM_SCRIPT="./run_pipeline_slurm.sh"

if [[ ! -x "${SLURM_SCRIPT}" ]]; then
  echo "[run_all_tokenizer_extensions] Missing SLURM script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

CONFIGS=(
  "tier1_12langs|configs/cola_tier1_12langs.yaml|4:00:00"
  "tier2_72langs|configs/cola_tier2_72langs.yaml|8:00:00"
  "tier3_200langs|configs/cola_tier3_200langs.yaml|12:00:00"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r tier config_path time_limit <<< "${entry}"
  if [[ ! -f "${config_path}" ]]; then
    echo "[run_all_tokenizer_extensions] Config for ${tier} not found: ${config_path}" >&2
    exit 1
  fi

  job_name="tokenizer_ext_${tier}"
  log_path="${LOG_ROOT}/${tier}_%j.log"
  echo "[run_all_tokenizer_extensions] Submitting ${tier} job with config ${config_path}"
  sbatch --job-name "${job_name}" --time "${time_limit}" --output "${log_path}" "${SLURM_SCRIPT}" "${config_path}"
done
