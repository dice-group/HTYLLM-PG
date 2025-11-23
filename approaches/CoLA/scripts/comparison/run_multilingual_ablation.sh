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
LANGUAGE_MAP=${LANGUAGE_MAP:-/scratch/hpc-prf-merlin/joel/moe-study/configs/moelpr_lang_map.json}
LANGUAGE_COLUMN=${LANGUAGE_COLUMN:-language}
TOKENIZED_BASE_DIR=${TOKENIZED_BASE_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter}

OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation}
WANDB_PROJECT=${WANDB_PROJECT:-htyllm-adapter-lpr}
WANDB_ENTITY=${WANDB_ENTITY:-}
default_wandb_group="multilingual-ablation"
if [[ -z "${WANDB_RUN_GROUP+x}" ]]; then
  WANDB_RUN_GROUP="${default_wandb_group}"
fi
SBATCH_ARGS=${SBATCH_ARGS:-}

required_paths=(
  DATASET_DIR
  LANGUAGE_MAP
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
export LANGUAGE_MAP LANGUAGE_COLUMN
export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ "${WANDB_RUN_GROUP}" == "${default_wandb_group}" ]]; then
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP}-${timestamp}"
fi

# Model entries: <tokenizer_dir>|<model_name_or_path>
MODEL_VARIANTS=(
  "llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B"
  # "llama-3.2-3B_tokenizer|meta-llama/Llama-3.2-3B"
  # "llama-3.1-8B_tokenizer|meta-llama/Llama-3.1-8B"
)

# Language tiers: <tier_id>|<language_count>|<subset_key>|<path_suffix>
LANGUAGE_TIERS=(
  "tier22|22|twenty_two_representatives_mediods|22_langs"
  # "tier95|95|ninty_five_representatives_mediods|95_langs"
  # "tier199|199|hundred_ninty_nine_representatives_mediods|199_langs"
)

# Hydra variants: <label>|<use_experts>|<lora_num>|<router_mode>|<prior_weight>|<bias_value>|<top_k>
HYDRA_VARIANTS=(
  "lora|False|1|learned|0.0|0.0|1"
  "hydra-flat|False|3|learned|0.0|0.0|1"
  # "hydra-route|True|3|learned|0.0|0.0|1"
  # "hydra-route-lpr|True|3|bias|0.3|2.5|1"
)

# CoLA variants: <label>|<use_experts>|<num_A>|<num_B>|<strategy>|<router_mode>|<prior_weight>|<bias_value>|<top_k>
COLA_VARIANTS=(
  "colaflat|False|1|3|fully|learned|0.0|0.0|1"
  "colaflat-rand|False|1|3|random_ba|learned|0.0|0.0|1"
  "colaexp|True|1|3|fully|learned|0.0|0.0|1"
  "colaexp-lpr|True|1|3|fully|bias|0.3|5.0|1"
  # "colafamily|True|1|3|fully|hard|0.0|0.0|1"
  "colafamily-lpr|True|1|3|fully|hard|0.3|5.0|1"
)

# Resource mappings
DEFAULT_GPU_TYPE=${DEFAULT_GPU_TYPE:-h100}
DEFAULT_GPU_COUNT=${DEFAULT_GPU_COUNT:-1}
DEFAULT_WALLTIME=${DEFAULT_WALLTIME:-08:00:00}
DEFAULT_PARTITION=${DEFAULT_PARTITION:-gpu}

declare -A MODEL_GPU_MAP=(
  ["meta-llama_Llama-3.2-1B"]=1
  ["meta-llama_Llama-3.2-3B"]=2
  ["meta-llama_Llama-3.1-8B"]=4
)

declare -A MODEL_GPU_TYPE_MAP=(
  ["meta-llama_Llama-3.2-1B"]="${DEFAULT_GPU_TYPE}"
  ["meta-llama_Llama-3.2-3B"]="${DEFAULT_GPU_TYPE}"
  ["meta-llama_Llama-3.1-8B"]="${DEFAULT_GPU_TYPE}"
)

declare -A MODEL_PARTITION_MAP=(
  ["meta-llama_Llama-3.2-1B"]="${DEFAULT_PARTITION}"
  ["meta-llama_Llama-3.2-3B"]="${DEFAULT_PARTITION}"
  ["meta-llama_Llama-3.1-8B"]="${DEFAULT_PARTITION}"
)

declare -A TIER_WALLTIME_MAP=(
  ["tier22"]="08:00:00"
  ["tier95"]="16:00:00"
  ["tier199"]="24:00:00"
)

ENABLE_LM_EVAL_LISTENER=${ENABLE_LM_EVAL_LISTENER:-false}
CHECKPOINT_LISTENER_SCRIPT=${CHECKPOINT_LISTENER_SCRIPT:-${REPO_ROOT}/scripts/checkpoint_listener.sh}
LM_EVAL_SCRIPT=${LM_EVAL_SCRIPT:-${REPO_ROOT}/scripts/lm_eval_checkpoint.sh}
LM_EVAL_TASKS=${LM_EVAL_TASKS:-"belebele,flores200,arc_challenge,mmlu,hellaswag"}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-auto}
LM_EVAL_POLL_INTERVAL=${LM_EVAL_POLL_INTERVAL:-300}
LM_EVAL_WANDB_PROJECT=${LM_EVAL_WANDB_PROJECT:-llama31_multilingual_eval_belebele}
LM_EVAL_WANDB_PREFIX=${LM_EVAL_WANDB_PREFIX:-lm_eval}
LM_EVAL_EXTRA_ARGS=${LM_EVAL_EXTRA_ARGS:-}
LISTENER_SBATCH_ARGS=${LISTENER_SBATCH_ARGS:-}

select_resources() {
  local -n out_ref=$1
  local model_slug=$2
  local tier_id=$3
  local gpu_type=${MODEL_GPU_TYPE_MAP[$model_slug]:-${DEFAULT_GPU_TYPE}}
  local gpu_count=${MODEL_GPU_MAP[$model_slug]:-${DEFAULT_GPU_COUNT}}
  local partition=${MODEL_PARTITION_MAP[$model_slug]:-${DEFAULT_PARTITION}}
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

for model_spec in "${MODEL_VARIANTS[@]}"; do
  IFS='|' read -r tokenizer_dir model_path <<<"${model_spec}"
  model_slug="$(sanitize "${model_path}")"

  for tier_spec in "${LANGUAGE_TIERS[@]}"; do
    IFS='|' read -r tier_id lang_count subset_key tier_path <<<"${tier_spec}"
    tier_slug="$(sanitize "${tier_id}")"
    tokenized_path="${TOKENIZED_BASE_DIR}/${tokenizer_dir}/${tier_path}"

    if [[ ! -d "${tokenized_path}" ]]; then
      echo "[ERROR] Tokenized path ${tokenized_path} not found for ${model_path} / ${tier_id}" >&2
      exit 1
    fi

    # Hydra/LoRA runs
    for variant_spec in "${HYDRA_VARIANTS[@]}"; do
      IFS='|' read -r label use_experts lora_num router_mode prior_weight bias_value top_k <<<"${variant_spec}"
      variant_slug="$(sanitize "${label}")"
      if [[ "${use_experts}" == "True" ]]; then
        hydra_num_experts="${lang_count}"
      else
        hydra_num_experts=1
      fi
      declare -a hydra_sbatch=()
      select_resources hydra_sbatch "${model_slug}" "${tier_id}"

      hydra_output="${OUTPUT_ROOT}/${model_slug}/${tier_slug}/hydra_${variant_slug}_${timestamp}"
      hydra_wandb="${label}-${model_slug}-${tier_id}-${timestamp}"
      hydra_log="hydra_${model_slug}_${tier_slug}_${variant_slug}"

      hydra_env=(
        "OUTPUT_DIR=${hydra_output}"
        "WANDB_NAME=${hydra_wandb}"
        "MODEL_NAME_OR_PATH=${model_path}"
        "TOKENIZED_PATH=${tokenized_path}"
        "LANGUAGE_ROUTER_MODE=${router_mode}"
        "LANGUAGE_PRIOR_WEIGHT=${prior_weight}"
        "LANGUAGE_BIAS_VALUE=${bias_value}"
        "USE_HYDRALORA_EXPERTS=${use_experts}"
        "HYDRALORA_NUM_EXPERTS=${hydra_num_experts}"
        "HYDRALORA_TOP_K=${top_k}"
        "LORA_NUM=${lora_num}"
        "MODEL_VARIANT=${model_slug}"
        "LANGUAGE_TIER=${tier_id}"
      )

      hydra_job=$(submit_job "Hydra-${label}-${model_slug}-${tier_id}" \
        "${COMPARISON_DIR}/hydralora_lpr_job.sh" "${hydra_log}" hydra_env hydra_sbatch)
      JOB_IDS["hydra-${label}-${model_slug}-${tier_id}"]="${hydra_job}"
      submit_listener_job "hydra-${label}-${model_slug}-${tier_id}" "${hydra_output}" "${model_path}"
    done

    # CoLA runs
    for variant_spec in "${COLA_VARIANTS[@]}"; do
      IFS='|' read -r label use_experts num_A num_B strategy router_mode prior_weight bias_value top_k <<<"${variant_spec}"
      variant_slug="$(sanitize "${label}")"
      if [[ "${use_experts}" == "True" ]]; then
        cola_num_experts="${lang_count}"
      else
        cola_num_experts=1
      fi
      declare -a cola_sbatch=()
      select_resources cola_sbatch "${model_slug}" "${tier_id}"

      cola_output="${OUTPUT_ROOT}/${model_slug}/${tier_slug}/cola_${variant_slug}_${timestamp}"
      cola_wandb="${label}-${model_slug}-${tier_id}-${timestamp}"
      cola_log="cola_${model_slug}_${tier_slug}_${variant_slug}"

      cola_env=(
        "OUTPUT_DIR=${cola_output}"
        "WANDB_NAME=${cola_wandb}"
        "MODEL_NAME_OR_PATH=${model_path}"
        "TOKENIZED_PATH=${tokenized_path}"
        "LANGUAGE_ROUTER_MODE=${router_mode}"
        "LANGUAGE_PRIOR_WEIGHT=${prior_weight}"
        "LANGUAGE_BIAS_VALUE=${bias_value}"
        "USE_COLA_EXPERTS=${use_experts}"
        "COLA_NUM_EXPERTS=${cola_num_experts}"
        "COLA_NUM_A=${num_A}"
        "COLA_NUM_B=${num_B}"
        "COLA_STRATEGY=${strategy}"
        "COLA_TOP_K=${top_k}"
        "MODEL_VARIANT=${model_slug}"
        "LANGUAGE_TIER=${tier_id}"
      )

      cola_job=$(submit_job "CoLA-${label}-${model_slug}-${tier_id}" \
        "${COMPARISON_DIR}/cola_lpr_job.sh" "${cola_log}" cola_env cola_sbatch)
      JOB_IDS["cola-${label}-${model_slug}-${tier_id}"]="${cola_job}"
      submit_listener_job "cola-${label}-${model_slug}-${tier_id}" "${cola_output}" "${model_path}"
    done
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
