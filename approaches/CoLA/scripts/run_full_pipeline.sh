#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/full_pipeline}"
mkdir -p "${LOG_ROOT}"

SAMPLE_SCRIPT="${REPO_ROOT}/sample_data/run_slurm_sampler.sh"
TOKENIZER_EXTENSION_SCRIPT="${REPO_ROOT}/data_prep/tokenizer_extension/run_pipeline_slurm.sh"
TOKENIZE_SCRIPT="${REPO_ROOT}/distributed_data_processor/tokenize_cola_tiers.sh"

PLAN_DIR="${REPO_ROOT}/sample_data/generate_sample_plan"
TOKENIZER_EXTENSION_ROOT="/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension"
TOKENIZED_OUTPUT_BASE="/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_tiers_tokenized"
BASE_TOKENIZER_SLUG="llama-3.1-8B_tokenizer"
declare -A EXT_TOKENIZER_SLUGS=(
  ["cola_tier1"]="cola_tier1_extended_tokenizer"
  ["cola_tier2"]="cola_tier2_extended_tokenizer"
)

SAMPLE_CONFIGS=(
  "tier1|${PLAN_DIR}/sampling_plan_tier1_12langs.csv|normal|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_12_tier_samples"
  "tier1|${PLAN_DIR}/sampling_plan_tier1_12langs.csv|tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/tokenizer_cola_12_tier_samples"
  "tier2|${PLAN_DIR}/sampling_plan_tier2_72langs.csv|normal|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_72_tier_samples"
  "tier2|${PLAN_DIR}/sampling_plan_tier2_72langs.csv|tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/tokenizer_cola_72_tier_samples"
  "tier3|${PLAN_DIR}/sampling_plan_tier3_200langs.csv|normal|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_200_tier_samples"
  "tier3|${PLAN_DIR}/sampling_plan_tier3_200langs.csv|tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/tokenizer_cola_200_tier_samples"
)

EXTENSION_CONFIGS=(
  "tier1|cola_tier1|${REPO_ROOT}/data_prep/tokenizer_extension/configs/cola_tier1_12langs.yaml|4:00:00"
  "tier2|cola_tier2|${REPO_ROOT}/data_prep/tokenizer_extension/configs/cola_tier2_72langs.yaml|8:00:00"
  "tier3|cola_tier3|${REPO_ROOT}/data_prep/tokenizer_extension/configs/cola_tier3_200langs.yaml|12:00:00"
)

MERGE_TIERS=(
  "tier1|cola_tier1"
  "tier2|cola_tier2"
  "tier3|cola_tier3"
)

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Required command '$1' is not available." >&2
    exit 1
  fi
}

prepare_output_dir() {
  local raw_path="$1"
  local dir
  dir="$(realpath -m "${raw_path}")"
  if [[ -z "${dir}" || "${dir}" == "/" || "${dir}" == "." || "${dir}" == ".." ]]; then
    echo "[ERROR] Unsafe path for prepare_output_dir: ${raw_path}" >&2
    exit 1
  fi
  if [[ -d "${dir}" ]]; then
    rm -rf "${dir}"
  fi
  mkdir -p "${dir}"
  echo "[pipeline] Prepared clean directory ${dir}"
}

prepare_tokenized_outputs() {
  local tier
  for tier in "${!EXT_TOKENIZER_SLUGS[@]}"; do
    local base_dir="${TOKENIZED_OUTPUT_BASE}/${BASE_TOKENIZER_SLUG}/${tier}"
    local base_ranks="${base_dir}_ranks"
    prepare_output_dir "${base_dir}"
    prepare_output_dir "${base_ranks}"

    local ext_slug="${EXT_TOKENIZER_SLUGS[${tier}]}"
    local ext_dir="${TOKENIZED_OUTPUT_BASE}/${ext_slug}"
    local ext_ranks="${ext_dir}_ranks"
    prepare_output_dir "${ext_dir}"
    prepare_output_dir "${ext_ranks}"
  done
}

wait_for_jobs() {
  local job_ids=("$@")
  local active=()
  if [[ ${#job_ids[@]} -eq 0 ]]; then
    return
  fi
  ensure_command squeue
  echo "[pipeline] waiting for jobs: ${job_ids[*]}"
  while true; do
    active=()
    for job in "${job_ids[@]}"; do
      if squeue -h -j "${job}" >/dev/null 2>&1; then
        active+=("${job}")
      fi
    done
    if [[ ${#active[@]} -eq 0 ]]; then
      break
    fi
    sleep 30
    job_ids=("${active[@]}")
  done
}

submit_sample_job() {
  local tier="$1"
  local plan="$2"
  local mode="$3"
  local output="$4"
  local dependency="$5"
  local flag=""
  [[ "${mode}" == "tokenizer" ]] && flag="--tokenizer-training"
  local job_name="fw2_${tier}_${mode}"
  local log_path="${LOG_ROOT}/sample_${tier}_${mode}_%j.log"
  local cmd=(sbatch --parsable --job-name "${job_name}" --output "${log_path}")
  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency "afterok:${dependency}")
  fi
  cmd+=(--export "ALL,OUTPUT_DIR=${output}" "${SAMPLE_SCRIPT}" "${plan}")
  [[ -n "${flag}" ]] && cmd+=("${flag}")
  "${cmd[@]}"
}

submit_extension_job() {
  local tier="$1"
  local config="$2"
  local time_limit="$3"
  local dependency="$4"
  local job_name="tokenizer_ext_${tier}"
  local log_path="${LOG_ROOT}/tokenizer_ext_${tier}_%j.log"
  local cmd=(sbatch --parsable --job-name "${job_name}" --output "${log_path}" --time "${time_limit}")
  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency "afterok:${dependency}")
  fi
  cmd+=("${TOKENIZER_EXTENSION_SCRIPT}" "${config}")
  "${cmd[@]}"
}

merge_model_and_tokenizer() {
  local tier="$1"
  local tier_dir="$2"
  local init_dir="${tier_dir}/initialized_model"
  local ext_tok_dir="${tier_dir}/extended_tokenizer"
  local merged_dir="${tier_dir}/merged_model"

  if [[ ! -d "${init_dir}" ]]; then
    echo "[merge] Missing initialized model for ${tier}: ${init_dir}" >&2
    return 1
  fi
  if [[ ! -d "${ext_tok_dir}" ]]; then
    echo "[merge] Missing extended tokenizer for ${tier}: ${ext_tok_dir}" >&2
    return 1
  fi

  rm -rf "${merged_dir}"
  mkdir -p "${merged_dir}"
  rsync -a "${init_dir}/" "${merged_dir}/"
  rsync -a "${ext_tok_dir}/" "${merged_dir}/"
  echo "[merge] Created merged model tokenizers for ${tier} at ${merged_dir}"
}

submit_tokenization_job() {
  local dependency="$1"
  local job_name="cola_tier_tokenization"
  local log_path="${LOG_ROOT}/cola_tier_tokenize_%j.log"
  local env_export="ALL,OUTPUT_BASE=${TOKENIZED_OUTPUT_BASE},BASE_TOKENIZER_SLUG=${BASE_TOKENIZER_SLUG}"
  local cmd=(sbatch --parsable --job-name "${job_name}" --output "${log_path}" --export "${env_export}")
  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency "afterok:${dependency}")
  fi
  cmd+=("${TOKENIZE_SCRIPT}")
  "${cmd[@]}"
}

main() {
  ensure_command sbatch
  ensure_command realpath
  ensure_command rsync
  local prev_job=""
  local sample_jobs=()
  for entry in "${SAMPLE_CONFIGS[@]}"; do
    IFS='|' read -r tier plan mode outdir <<<"${entry}"
    if [[ ! -f "${plan}" ]]; then
      echo "[pipeline] sample plan not found: ${plan}" >&2
      exit 1
    fi
    prepare_output_dir "${outdir}"
    local job_id
    job_id=$(submit_sample_job "${tier}" "${plan}" "${mode}" "${outdir}" "${prev_job}")
    sample_jobs+=("${job_id}")
    prev_job="${job_id}"
    echo "[pipeline] submitted sample job ${job_id} (${tier}-${mode})"
  done

  wait_for_jobs "${sample_jobs[@]}"

  local extension_jobs=()
  local last_sample_job="${sample_jobs[$(( ${#sample_jobs[@]} - 1 ))]}"
  prev_job="${last_sample_job}"
  for entry in "${EXTENSION_CONFIGS[@]}"; do
    IFS='|' read -r tier slug config time_limit <<<"${entry}"
    if [[ ! -f "${config}" ]]; then
      echo "[pipeline] extension config not found: ${config}" >&2
      exit 1
    fi
    prepare_output_dir "${TOKENIZER_EXTENSION_ROOT}/${slug}"
    local job_id
    job_id=$(submit_extension_job "${tier}" "${config}" "${time_limit}" "${prev_job}")
    extension_jobs+=("${job_id}")
    prev_job="${job_id}"
    echo "[pipeline] submitted extension job ${job_id} (${tier})"
  done

  wait_for_jobs "${extension_jobs[@]}"

  local last_extension_job="${extension_jobs[$(( ${#extension_jobs[@]} - 1 ))]}"

  for entry in "${MERGE_TIERS[@]}"; do
    IFS='|' read -r tier slug <<<"${entry}"
    merge_model_and_tokenizer "${tier}" "${TOKENIZER_EXTENSION_ROOT}/${slug}"
  done

  prepare_tokenized_outputs
  local tokenize_job
  tokenize_job=$(submit_tokenization_job "${last_extension_job}")
  echo "[pipeline] submitted tokenization wrapper job ${tokenize_job}"
  wait_for_jobs "${tokenize_job}"
}

main "$@"
