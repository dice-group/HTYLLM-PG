#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARISON_DIR="${SCRIPT_DIR}"
CALL_DIR="$(pwd)"
LOG_DIR="${LOG_DIR:-${CALL_DIR}/logs/lpr_ablation}"

cd "${REPO_ROOT}"

CONDA_BASE=${CONDA_BASE:-/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3}
CONDA_ENV=${CONDA_ENV:-cola_llama_factory}
MODULE_INIT=${MODULE_INIT:-module purge && module load toolchain/foss/2024a system/CUDA/12.6.0 lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B}
TOKENIZED_PATH=${TOKENIZED_PATH:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/5_langs}
DATASET_NAME=${DATASET_NAME:-c4}
DATASET_DIR=${DATASET_DIR:-./LLaMA-Factory/data}
LANGUAGE_MAP=${LANGUAGE_MAP:-/scratch/hpc-prf-merlin/joel/moe-study/configs/moelpr_lang_map.json}
LANGUAGE_COLUMN=${LANGUAGE_COLUMN:-language}

OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/hpc-prf-merlin/project_data/moe_study/lpr_ablation}
WANDB_PROJECT=${WANDB_PROJECT:-htyllm-adapter-lpr}
WANDB_ENTITY=${WANDB_ENTITY:-}
default_wandb_group="lpr-ablation"
if [[ -z "${WANDB_RUN_GROUP+x}" ]]; then
  WANDB_RUN_GROUP="${default_wandb_group}"
fi
SBATCH_ARGS=${SBATCH_ARGS:-}

required_paths=(
  TOKENIZED_PATH
  DATASET_DIR
  LANGUAGE_MAP
)

for path_var in "${required_paths[@]}"; do
  if [[ ! -e "${!path_var}" ]]; then
    echo "[ERROR] ${path_var} points to ${!path_var}, which does not exist." >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

export CONDA_BASE CONDA_ENV MODULE_INIT
export MODEL_NAME_OR_PATH TOKENIZED_PATH DATASET_NAME DATASET_DIR
export LANGUAGE_MAP LANGUAGE_COLUMN
export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ "${WANDB_RUN_GROUP}" == "${default_wandb_group}" ]]; then
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP}-${timestamp}"
fi

declare -a experiments=(
  "baseline|learned|0.0|0.0"
  "soft_prior|learned|0.3|0.0"
  "bias_prior|bias|0.3|2.5"
  "hard_prior|hard|0.1|0.0"
)

COLA_EXPERT_COUNT=${COLA_EXPERT_COUNT:-4}
COLA_FLAT_EXPERT_COUNT=${COLA_FLAT_EXPERT_COUNT:-1}
HYDRALORA_NUM_EXPERTS=${HYDRALORA_NUM_EXPERTS:-1}
HYDRALORA_TOP_K=${HYDRALORA_TOP_K:-1}
HYDRALORA_LORA_NUM=${HYDRALORA_LORA_NUM:-4}
USE_HYDRALORA_EXPERTS=${USE_HYDRALORA_EXPERTS:-False}

echo "[INFO] Launching Language Prior ablation runs into ${OUTPUT_ROOT}"

submit_job() {
  local name=$1
  local script=$2
  local log_prefix=$3
  shift 3
  local env_assignments=("$@")
  local cmd=(env)
  for assignment in "${env_assignments[@]}"; do
    cmd+=("${assignment}")
  done
  cmd+=("sbatch" "--output=${LOG_DIR}/${log_prefix}_%j.log")
  if [[ -n "${SBATCH_ARGS:-}" ]]; then
    cmd+=(${SBATCH_ARGS})
  fi
  cmd+=("${script}")
  local output
  if ! output=$("${cmd[@]}"); then
    echo "[ERROR] Failed to submit ${name} job." >&2
    exit 1
  fi
  if [[ "${output}" != *"Submitted batch job"* ]]; then
    echo "${output}" >&2
    echo "[ERROR] sbatch did not acknowledge submission for ${name}." >&2
    exit 1
  fi
  local job_id
  job_id=$(awk '{print $4}' <<<"${output}")
  echo "[INFO] Submitted ${name} job ${job_id}"
  printf "%s" "${job_id}"
}

declare -A JOB_IDS=()
for spec in "${experiments[@]}"; do
  IFS='|' read -r label router_mode prior_weight bias_value <<<"${spec}"

  # CoLA with experts (MoE)
  cola_exp_output="${OUTPUT_ROOT}/colaexp_${label}_${timestamp}"
  cola_exp_wandb="colaexp-${label}-${timestamp}"
  cola_exp_log="colaexp_${label}"
  cola_exp_env=(
    "OUTPUT_DIR=${cola_exp_output}"
    "WANDB_NAME=${cola_exp_wandb}"
    "LANGUAGE_ROUTER_MODE=${router_mode}"
    "LANGUAGE_PRIOR_WEIGHT=${prior_weight}"
    "LANGUAGE_BIAS_VALUE=${bias_value}"
    "USE_COLA_EXPERTS=True"
    "COLA_NUM_EXPERTS=${COLA_EXPERT_COUNT}"
  )
  cola_exp_job=$(submit_job "CoLA-Experts-${label}" "${COMPARISON_DIR}/cola_lpr_job.sh" "${cola_exp_log}" "${cola_exp_env[@]}")
  JOB_IDS["colaexp-${label}"]="${cola_exp_job}"

  # CoLA without experts (baseline CoLA)
  cola_flat_output="${OUTPUT_ROOT}/colaflat_${label}_${timestamp}"
  cola_flat_wandb="colaflat-${label}-${timestamp}"
  cola_flat_log="colaflat_${label}"
  cola_flat_env=(
    "OUTPUT_DIR=${cola_flat_output}"
    "WANDB_NAME=${cola_flat_wandb}"
    "LANGUAGE_ROUTER_MODE=${router_mode}"
    "LANGUAGE_PRIOR_WEIGHT=${prior_weight}"
    "LANGUAGE_BIAS_VALUE=${bias_value}"
    "USE_COLA_EXPERTS=False"
    "COLA_NUM_EXPERTS=${COLA_FLAT_EXPERT_COUNT}"
  )
  cola_flat_job=$(submit_job "CoLA-Flat-${label}" "${COMPARISON_DIR}/cola_lpr_job.sh" "${cola_flat_log}" "${cola_flat_env[@]}")
  JOB_IDS["colaflat-${label}"]="${cola_flat_job}"

  # HydraLoRA baseline
  hydra_output="${OUTPUT_ROOT}/hydra_${label}_${timestamp}"
  hydra_wandb="hydra-${label}-${timestamp}"
  hydra_log="hydra_${label}"
  hydra_env=(
    "OUTPUT_DIR=${hydra_output}"
    "WANDB_NAME=${hydra_wandb}"
    "LANGUAGE_ROUTER_MODE=${router_mode}"
    "LANGUAGE_PRIOR_WEIGHT=${prior_weight}"
    "LANGUAGE_BIAS_VALUE=${bias_value}"
    "USE_HYDRALORA_EXPERTS=${USE_HYDRALORA_EXPERTS}"
    "HYDRALORA_NUM_EXPERTS=${HYDRALORA_NUM_EXPERTS}"
    "HYDRALORA_TOP_K=${HYDRALORA_TOP_K}"
    "LORA_NUM=${HYDRALORA_LORA_NUM}"
  )
  hydra_job=$(submit_job "HydraLoRA-${label}" "${COMPARISON_DIR}/hydralora_lpr_job.sh" "${hydra_log}" "${hydra_env[@]}")
  JOB_IDS["hydra-${label}"]="${hydra_job}"
done

cat <<SUMMARY
========================================
Submitted Language-Prior ablation jobs at ${timestamp}
$(for label in "${!JOB_IDS[@]}"; do printf "  %-18s : %s\n" "${label}" "${JOB_IDS[${label}]}"; done | sort)

Outputs stored under ${OUTPUT_ROOT}
Monitor metrics in W&B project ${WANDB_PROJECT:-moe-study-lpr}
========================================
SUMMARY
