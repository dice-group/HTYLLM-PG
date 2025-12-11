#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARISON_DIR="${SCRIPT_DIR}"
CALL_DIR="$(pwd)"
LOG_DIR="${LOG_DIR:-${CALL_DIR}/logs/multilingual_ablation}"

cd "${REPO_ROOT}"

CONDA_BASE=${CONDA_BASE:-/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3}
CONDA_ENV=${CONDA_ENV:-cola_llama_factory}
MODULE_INIT=${MODULE_INIT:-module purge && module load toolchain/foss/2024a system/CUDA/12.6.0 lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0}

DATASET_NAME=${DATASET_NAME:-c4}
DATASET_DIR=${DATASET_DIR:-./LLaMA-Factory/data}
LANGUAGE_COLUMN=${LANGUAGE_COLUMN:-language}
TOKENIZED_BASE_DIR=${TOKENIZED_BASE_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_tiers_tokenized}

OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation}
WANDB_PROJECT=${WANDB_PROJECT:-htyllm-adapter-lpr-12_72_lang_tier}
WANDB_ENTITY=${WANDB_ENTITY:-}
default_wandb_group="multilingual-ablation"
if [[ -z "${WANDB_RUN_GROUP+x}" ]]; then
  WANDB_RUN_GROUP="${default_wandb_group}"
fi
SBATCH_ARGS=${SBATCH_ARGS:-}

required_paths=(
  DATASET_DIR
  TOKENIZED_BASE_DIR
)

for path_var in "${required_paths[@]}"; do
  if [[ ! -e "${!path_var}" ]]; then
    echo "[ERROR] ${path_var} points to ${!path_var}, which does not exist." >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

export CONDA_BASE CONDA_ENV MODULE_INIT
export DATASET_NAME DATASET_DIR
export LANGUAGE_COLUMN
export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ "${WANDB_RUN_GROUP}" == "${default_wandb_group}" ]]; then
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP}-${timestamp}"
fi

# Language tiers: <tier_id>|<language_count>|<language_map_path>|<tokenized_path>|<model_path>
LANGUAGE_TIERS=(
  "tier12|12|${REPO_ROOT}/tools/two_stage_clustering/12_tier_language_groupings.json|${TOKENIZED_BASE_DIR}/cola_tier1_extended_tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier1/merged_model"
  "tier12_base|12|${REPO_ROOT}/tools/two_stage_clustering/12_tier_language_groupings.json|${TOKENIZED_BASE_DIR}/llama-3.1-8B_tokenizer/cola_tier1|meta-llama/Llama-3.1-8B"      # i hope this will sparately show the effect of the extended tokenizer and weight initialization
  "tier72|72|${REPO_ROOT}/tools/two_stage_clustering/72_tier_language_groupings.json|${TOKENIZED_BASE_DIR}/cola_tier2_extended_tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier2/merged_model"
  # "tier200|200|${REPO_ROOT}/tools/two_stage_clustering/200_tier_language_groupings.json|${TOKENIZED_BASE_DIR}/cola_tier3_extended_tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier3/merged_model"
)

# Hydra variants: <label>|<use_experts>|<lora_num>|<router_mode>|<prior_weight>|<bias_value>|<top_k>|<guidance_scope>
HYDRA_VARIANTS=(
  "hydra-lora|False|1|learned|0.0|0.0|1|none"
  "hydra-flat|False|3|learned|0.0|0.0|1|none"
  "hydra-exp-lpr|True|3|learned|0.1|0.0|1|all"
)

# CoLA variants: <label>|<use_experts>|<num_A>|<num_B>|<strategy>|<router_mode>|<prior_weight>|<bias_value>|<top_k>|<guidance_scope>
COLA_VARIANTS=(
  "colaflat|False|1|3|fully|learned|0.0|0.0|1|none"
  "colaexp-lpr|True|1|3|fully|learned|0.1|0.0|1|all"
)

# Resource mappings
DEFAULT_GPU_TYPE=${DEFAULT_GPU_TYPE:-h100}
DEFAULT_GPU_COUNT=${DEFAULT_GPU_COUNT:-1}
DEFAULT_WALLTIME=${DEFAULT_WALLTIME:-12:00:00}
DEFAULT_PARTITION=${DEFAULT_PARTITION:-gpu}

declare -A TIER_WALLTIME_MAP=(
  ["tier12"]="12:00:00"
  ["tier12_base"]="12:00:00"
  ["tier72"]="18:00:00"
  ["tier200"]="24:00:00"
)

declare -A TIER_GPU_COUNT_MAP=(
  ["tier12"]=2
  ["tier12_base"]=2
  ["tier72"]=4
  ["tier200"]=8
)

declare -A TIER_GPU_TYPE_MAP=(
  ["tier12"]="${DEFAULT_GPU_TYPE}"
  ["tier12_base"]="${DEFAULT_GPU_TYPE}"
  ["tier72"]="${DEFAULT_GPU_TYPE}"
  ["tier200"]="${DEFAULT_GPU_TYPE}"
)

declare -A TIER_PARTITION_MAP=(
  ["tier12"]="${DEFAULT_PARTITION}"
  ["tier12_base"]="${DEFAULT_PARTITION}"
  ["tier72"]="${DEFAULT_PARTITION}"
  ["tier200"]="${DEFAULT_PARTITION}"
)

ENABLE_LM_EVAL_LISTENER=${ENABLE_LM_EVAL_LISTENER:-false}
CHECKPOINT_LISTENER_SCRIPT=${CHECKPOINT_LISTENER_SCRIPT:-${REPO_ROOT}/scripts/checkpoint_listener.sh}
LM_EVAL_SCRIPT=${LM_EVAL_SCRIPT:-${REPO_ROOT}/scripts/lm_eval_checkpoint.sh}
LM_EVAL_TASK_FILE=${LM_EVAL_TASK_FILE:-${REPO_ROOT}/configs/lm_eval_tasks.txt}
if [[ -z "${LM_EVAL_TASKS:-}" ]]; then
  if [[ -f "${LM_EVAL_TASK_FILE}" ]]; then
    LM_EVAL_TASKS=$(paste -sd, "${LM_EVAL_TASK_FILE}")
  else
    LM_EVAL_TASKS="belebele,flores200,arc_challenge,mmlu,hellaswag"
  fi
fi
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-auto}
LM_EVAL_POLL_INTERVAL=${LM_EVAL_POLL_INTERVAL:-300}
LM_EVAL_WANDB_PROJECT=${LM_EVAL_WANDB_PROJECT:-llama31_multilingual_eval_belebele}
LM_EVAL_WANDB_PREFIX=${LM_EVAL_WANDB_PREFIX:-lm_eval}
LM_EVAL_EXTRA_ARGS=${LM_EVAL_EXTRA_ARGS:-}
LISTENER_SBATCH_ARGS=${LISTENER_SBATCH_ARGS:-}

select_resources() {
  local -n out_ref=$1
  local tier_id=$2
  local gpu_type=${TIER_GPU_TYPE_MAP[$tier_id]:-${DEFAULT_GPU_TYPE}}
  local gpu_count=${TIER_GPU_COUNT_MAP[$tier_id]:-${DEFAULT_GPU_COUNT}}
  local partition=${TIER_PARTITION_MAP[$tier_id]:-${DEFAULT_PARTITION}}
  local walltime=${TIER_WALLTIME_MAP[$tier_id]:-${DEFAULT_WALLTIME}}
  out_ref=(
    "--partition=${partition}"
    "--gres=gpu:${gpu_type}:${gpu_count}"
    "--time=${walltime}"
  )
}

echo "[INFO] Launching multilingual ablation runs into ${OUTPUT_ROOT}"

submit_job() {
  local name=$1
  local script=$2
  local log_prefix=$3
  local env_array_name=$4
  local sbatch_array_name=$5
  shift 5
  local -n env_assignments=$env_array_name
  local -n sbatch_assignments=$sbatch_array_name
  local cmd=(env)
  for assignment in "${env_assignments[@]}"; do
    cmd+=("${assignment}")
  done
  cmd+=("sbatch" "--output=${LOG_DIR}/${log_prefix}_%j.log")
  if [[ -n "${SBATCH_ARGS:-}" ]]; then
    cmd+=(${SBATCH_ARGS})
  fi
  for arg in "${sbatch_assignments[@]}"; do
    if [[ -n "${arg}" ]]; then
      cmd+=("${arg}")
    fi
  done
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

submit_listener_job() {
  local listener_label=$1
  local watch_dir=$2
  local model_path=$3
  if [[ "${ENABLE_LM_EVAL_LISTENER}" != "true" ]]; then
    return
  fi
  if [[ ! -f "${CHECKPOINT_LISTENER_SCRIPT}" ]]; then
    echo "[WARN] CHECKPOINT_LISTENER_SCRIPT not found at ${CHECKPOINT_LISTENER_SCRIPT}; skipping listener launch" >&2
    return
  fi
  if [[ ! -f "${LM_EVAL_SCRIPT}" ]]; then
    echo "[WARN] LM_EVAL_SCRIPT not found at ${LM_EVAL_SCRIPT}; skipping listener launch" >&2
    return
  fi
  local log_path="${LOG_DIR}/${listener_label}_listener_%j.log"
  local cmd=(sbatch "--output=${log_path}")
  if [[ -n "${LISTENER_SBATCH_ARGS}" ]]; then
    cmd+=(${LISTENER_SBATCH_ARGS})
  fi
  cmd+=(
    "${CHECKPOINT_LISTENER_SCRIPT}"
    --watch-dir "${watch_dir}"
    --eval-script "${LM_EVAL_SCRIPT}"
    --tokenizer "${model_path}"
    --tasks "${LM_EVAL_TASKS}"
    --batch-size "${LM_EVAL_BATCH_SIZE}"
    --poll-interval "${LM_EVAL_POLL_INTERVAL}"
    --wandb-project "${LM_EVAL_WANDB_PROJECT}"
    --wandb-prefix "${LM_EVAL_WANDB_PREFIX}_${listener_label}"
  )
  if [[ -n "${LM_EVAL_EXTRA_ARGS}" ]]; then
    cmd+=(--extra-args "${LM_EVAL_EXTRA_ARGS}")
  fi
  local output
  if ! output=$("${cmd[@]}"); then
    echo "[WARN] Failed to submit listener for ${listener_label}" >&2
    return
  fi
  if [[ "${output}" != *"Submitted batch job"* ]]; then
    echo "[WARN] Listener submission returned unexpected output: ${output}" >&2
    return
  fi
  local job_id
  job_id=$(awk '{print $4}' <<<"${output}")
  echo "[INFO] Submitted checkpoint listener ${job_id} for ${listener_label}"
  LISTENER_JOB_IDS["listener-${listener_label}"]="${job_id}"
}

sanitize() {
  echo "$1" | tr '/ ' '__'
}

declare -A JOB_IDS=()
declare -A LISTENER_JOB_IDS=()

for tier_spec in "${LANGUAGE_TIERS[@]}"; do
  IFS='|' read -r tier_id lang_count tier_map tokenized_path model_path <<<"${tier_spec}"
  tier_slug="$(sanitize "${tier_id}")"
  if [[ "${tokenized_path}" != /* ]]; then
    tokenized_path="${TOKENIZED_BASE_DIR}/${tokenized_path}"
  fi
  if [[ ! -d "${tokenized_path}" ]]; then
    echo "[ERROR] Tokenized path ${tokenized_path} not found for ${tier_id}" >&2
    exit 1
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "[ERROR] Model path ${model_path} not found for ${tier_id}" >&2
    exit 1
  fi

  run_lora_only=false
  if [[ "${tier_id}" == "tier12_base" ]]; then
    run_lora_only=true
  fi

  # Hydra/LoRA runs
  for variant_spec in "${HYDRA_VARIANTS[@]}"; do
    IFS='|' read -r label use_experts lora_num router_mode prior_weight bias_value top_k guidance_scope <<<"${variant_spec}"
    variant_slug="$(sanitize "${label}")"
    if [[ "${run_lora_only}" == "true" && "${label}" != "hydra-lora" ]]; then
      continue
    fi
    if [[ "${use_experts}" == "True" ]]; then
      hydra_num_experts="${lang_count}"
    else
      hydra_num_experts=1
    fi
    declare -a hydra_sbatch=()
    select_resources hydra_sbatch "${tier_id}"
    tier_gpu_count=${TIER_GPU_COUNT_MAP[$tier_id]:-${DEFAULT_GPU_COUNT}}
    accelerate_config=""
    if [[ "${tier_gpu_count}" -ge 2 ]]; then
      if [[ "${tier_gpu_count}" -ge 4 ]]; then
        accelerate_config="${REPO_ROOT}/LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml"
      else
        accelerate_config="${REPO_ROOT}/LLaMA-Factory/examples/accelerate/fsdp_2gpu_config.yaml"
      fi
    fi

    hydra_descriptor="${tier_id}_${label}_${router_mode}_g${prior_weight}"
    hydra_output="${OUTPUT_ROOT}/${tier_slug}/hydra_${variant_slug}_${timestamp}"
    hydra_wandb="${hydra_descriptor}_${timestamp}"
    hydra_log="hydra_${tier_slug}_${variant_slug}"

    hydra_env=(
      "OUTPUT_DIR=${hydra_output}"
      "WANDB_NAME=${hydra_wandb}"
      "MODEL_NAME_OR_PATH=${model_path}"
      "TOKENIZED_PATH=${tokenized_path}"
      "LANGUAGE_MAP=${tier_map}"
      "LANGUAGE_ROUTER_MODE=${router_mode}"
      "LANGUAGE_PRIOR_WEIGHT=${prior_weight}"
      "LANGUAGE_BIAS_VALUE=${bias_value}"
      "LANGUAGE_GUIDANCE_SCOPE=${guidance_scope}"
      "USE_HYDRALORA_EXPERTS=${use_experts}"
      "HYDRALORA_NUM_EXPERTS=${hydra_num_experts}"
      "HYDRALORA_TOP_K=${top_k}"
      "LORA_NUM=${lora_num}"
      "MODEL_VARIANT=${tier_id}"
      "LANGUAGE_TIER=${tier_id}"
      "WANDB_TAGS=adapter:hydra,variant:${label},tier:${tier_id},mode:${router_mode},gamma:${prior_weight}"
    )
    if [[ -n "${accelerate_config}" ]]; then
      hydra_env+=("ACCELERATE_CONFIG_FILE=${accelerate_config}")
    fi

    hydra_job=$(submit_job "Hydra-${label}-${tier_id}" \
      "${COMPARISON_DIR}/hydralora_lpr_job.sh" "${hydra_log}" hydra_env hydra_sbatch)
    JOB_IDS["hydra-${label}-${tier_id}"]="${hydra_job}"
    submit_listener_job "hydra-${label}-${tier_id}" "${hydra_output}" "${model_path}"
  done

  # CoLA runs
  if [[ "${run_lora_only}" == "true" ]]; then
    continue
  fi
  for variant_spec in "${COLA_VARIANTS[@]}"; do
    IFS='|' read -r label use_experts num_A num_B strategy router_mode prior_weight bias_value top_k guidance_scope <<<"${variant_spec}"
    variant_slug="$(sanitize "${label}")"
    if [[ "${use_experts}" == "True" ]]; then
      cola_num_experts="${lang_count}"
    else
      cola_num_experts=1
    fi
    declare -a cola_sbatch=()
    select_resources cola_sbatch "${tier_id}"
    tier_gpu_count=${TIER_GPU_COUNT_MAP[$tier_id]:-${DEFAULT_GPU_COUNT}}
    accelerate_config=""
    if [[ "${tier_gpu_count}" -ge 2 ]]; then
      if [[ "${tier_gpu_count}" -ge 4 ]]; then
        accelerate_config="${REPO_ROOT}/LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml"
      else
        accelerate_config="${REPO_ROOT}/LLaMA-Factory/examples/accelerate/fsdp_2gpu_config.yaml"
      fi
    fi

    cola_descriptor="${tier_id}_${label}_${router_mode}_g${prior_weight}"
    cola_output="${OUTPUT_ROOT}/${tier_slug}/cola_${variant_slug}_${timestamp}"
    cola_wandb="${cola_descriptor}_${timestamp}"
    cola_log="cola_${tier_slug}_${variant_slug}"

    cola_env=(
      "OUTPUT_DIR=${cola_output}"
      "WANDB_NAME=${cola_wandb}"
      "MODEL_NAME_OR_PATH=${model_path}"
      "TOKENIZED_PATH=${tokenized_path}"
      "LANGUAGE_MAP=${tier_map}"
      "LANGUAGE_ROUTER_MODE=${router_mode}"
      "LANGUAGE_PRIOR_WEIGHT=${prior_weight}"
      "LANGUAGE_BIAS_VALUE=${bias_value}"
      "LANGUAGE_GUIDANCE_SCOPE=${guidance_scope}"
      "USE_COLA_EXPERTS=${use_experts}"
      "COLA_NUM_EXPERTS=${cola_num_experts}"
      "COLA_NUM_A=${num_A}"
      "COLA_NUM_B=${num_B}"
      "COLA_STRATEGY=${strategy}"
      "COLA_TOP_K=${top_k}"
      "MODEL_VARIANT=${tier_id}"
      "LANGUAGE_TIER=${tier_id}"
      "WANDB_TAGS=adapter:cola,variant:${label},tier:${tier_id},mode:${router_mode},gamma:${prior_weight}"
    )
    if [[ -n "${accelerate_config}" ]]; then
      cola_env+=("ACCELERATE_CONFIG_FILE=${accelerate_config}")
    fi

    cola_job=$(submit_job "CoLA-${label}-${tier_id}" \
      "${COMPARISON_DIR}/cola_lpr_job.sh" "${cola_log}" cola_env cola_sbatch)
    JOB_IDS["cola-${label}-${tier_id}"]="${cola_job}"
    submit_listener_job "cola-${label}-${tier_id}" "${cola_output}" "${model_path}"
  done
done

cat <<SUMMARY
========================================
Submitted multilingual ablation jobs at ${timestamp}
$(for key in "${!JOB_IDS[@]}"; do printf "  %-32s : %s\n" "${key}" "${JOB_IDS[${key}]}"; done | sort)
$(for key in "${!LISTENER_JOB_IDS[@]}"; do printf "  %-32s : %s\n" "${key}" "${LISTENER_JOB_IDS[${key}]}"; done | sort)

Outputs stored under ${OUTPUT_ROOT}
Tokenized corpora read from ${TOKENIZED_BASE_DIR}
Monitor metrics in W&B project ${WANDB_PROJECT}
========================================
SUMMARY
