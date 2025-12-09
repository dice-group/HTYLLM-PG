#!/usr/bin/env bash
set -euo pipefail

# Submits tokenizer-extension jobs for all CoLA tiers (12, 72, 200 languages).
# Each job reuses the tokenizer-extension pipeline and points it to the sampled
# tokenizer-training datasets generated via sample_data/run_all_samplers.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${SCRIPT_DIR}/logs/tokenize_extension"
CONFIG_DIR="${SCRIPT_DIR}/configs"
SLURM_SCRIPT="${SCRIPT_DIR}/run_pipeline_slurm.sh"

if [[ ! -x "${SLURM_SCRIPT}" ]]; then
  echo "[run_all_tokenizer_extensions] Missing or non-executable SLURM script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

CONFIGS=(
  "tier1_12langs|${CONFIG_DIR}/cola_tier1_12langs.yaml"
  "tier2_72langs|${CONFIG_DIR}/cola_tier2_72langs.yaml"
  "tier3_200langs|${CONFIG_DIR}/cola_tier3_200langs.yaml"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r tier config_path <<< "${entry}"
  if [[ ! -f "${config_path}" ]]; then
    echo "[run_all_tokenizer_extensions] Config for ${tier} not found: ${config_path}" >&2
    exit 1
  fi

  job_name="tokenizer_ext_${tier}"
  log_path="${LOG_ROOT}/${tier}_%j.log"
  echo "[run_all_tokenizer_extensions] Submitting ${tier} job with config ${config_path}"
  sbatch \
    --chdir "${SCRIPT_DIR}" \
    --job-name "${job_name}" \
    --output "${log_path}" \
    "${SLURM_SCRIPT}" "${config_path}"
done
