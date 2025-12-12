#!/usr/bin/env bash
# Launch tokenization + merge jobs for the CoLA Tier-12 and Tier-72 corpora
# with both the original Llama-3.1-8B tokenizer and the tier-specific extended tokenizers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/tokenize_and_merge_pipeline.sh"

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "[ERROR] Missing pipeline script at ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

# Paths to the language-sharded samples for each tier.
COLA_TIER1_SAMPLE_DIR=${COLA_TIER1_SAMPLE_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_12_tier_samples/sharded_samples}
COLA_TIER2_SAMPLE_DIR=${COLA_TIER2_SAMPLE_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_72_tier_samples/sharded_samples}

# Tokenizer identifiers.
BASE_TOKENIZER_NAME=${BASE_TOKENIZER_NAME:-meta-llama/Llama-3.1-8B}
BASE_TOKENIZER_SLUG=${BASE_TOKENIZER_SLUG:-llama-3.1-8B_tokenizer}
COLA_TIER1_EXT_TOKENIZER=${COLA_TIER1_EXT_TOKENIZER:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier1/extended_tokenizer}
COLA_TIER2_EXT_TOKENIZER=${COLA_TIER2_EXT_TOKENIZER:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier2/extended_tokenizer}

# Where to write the merged HF datasets (the pipeline also writes *_ranks/ siblings).
OUTPUT_BASE=${OUTPUT_BASE:-/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_tiers_tokenized}

# Resource knobs shared across runs (override via env vars if needed).
DEFAULT_NUM_PROC=${NUM_PROC:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
MEM_PER_TASK=${MEM_PER_TASK:-32G}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MERGE_CPUS=${MERGE_CPUS:-8}
MERGE_MEM=${MERGE_MEM:-96G}
MERGE_TIME=${MERGE_TIME:-06:00:00}
EVAL_FRACTION=${EVAL_FRACTION:-0.02}
EVAL_SEED=${EVAL_SEED:-42}
JOB_PREFIX_BASE=${JOB_PREFIX_BASE:-cola_tier_tok}

# Tier-specific parallelism defaults (override with COLA_TIER{1,2}_NUM_RANKS).
COLA_TIER1_NUM_RANKS=${COLA_TIER1_NUM_RANKS:-12}
COLA_TIER2_NUM_RANKS=${COLA_TIER2_NUM_RANKS:-72}
COLA_TIER1_NUM_PROC=${COLA_TIER1_NUM_PROC:-${DEFAULT_NUM_PROC}}
COLA_TIER2_NUM_PROC=${COLA_TIER2_NUM_PROC:-${DEFAULT_NUM_PROC}}

declare -A TIER_SAMPLE_DIRS=(
  ["cola_tier1"]="${COLA_TIER1_SAMPLE_DIR}"
  ["cola_tier2"]="${COLA_TIER2_SAMPLE_DIR}"
)
declare -A TIER_NUM_RANKS=(
  ["cola_tier1"]="${COLA_TIER1_NUM_RANKS}"
  ["cola_tier2"]="${COLA_TIER2_NUM_RANKS}"
)
declare -A TIER_EXT_TOKENIZERS=(
  ["cola_tier1"]="${COLA_TIER1_EXT_TOKENIZER}"
  ["cola_tier2"]="${COLA_TIER2_EXT_TOKENIZER}"
)
declare -A TIER_EXT_SLUGS=(
  ["cola_tier1"]="cola_tier1_extended_tokenizer"
  ["cola_tier2"]="cola_tier2_extended_tokenizer"
)
declare -A TIER_NUM_PROC=(
  ["cola_tier1"]="${COLA_TIER1_NUM_PROC}"
  ["cola_tier2"]="${COLA_TIER2_NUM_PROC}"
)
declare -a TIER_ORDER=("cola_tier1" "cola_tier2")

submit_pipeline() {
  local shard_dir=$1
  local tokenizer=$2
  local output_root=$3
  local job_prefix=$4
  local num_ranks=$5
  local num_proc=$6

  if [[ ! -d "${shard_dir}" ]]; then
    echo "[ERROR] Shard directory ${shard_dir} not found" >&2
    exit 1
  fi

  local cmd=(
    "${PIPELINE_SCRIPT}"
    --shard-dir "${shard_dir}"
    --tokenizer "${tokenizer}"
    --output-root "${output_root}"
    --num-ranks "${num_ranks}"
    --num-proc "${num_proc}"
    --cpus-per-task "${CPUS_PER_TASK}"
    --mem "${MEM_PER_TASK}"
    --time "${TIME_LIMIT}"
    --merge-cpus "${MERGE_CPUS}"
    --merge-mem "${MERGE_MEM}"
    --merge-time "${MERGE_TIME}"
    --eval-fraction "${EVAL_FRACTION}"
    --eval-seed "${EVAL_SEED}"
    --job-prefix "${job_prefix}"
  )

  echo "[INFO] Submitting ${job_prefix}: tokenizer=${tokenizer}, output=${output_root}"
  "${cmd[@]}"
}

for tier in "${TIER_ORDER[@]}"; do
  shard_dir="${TIER_SAMPLE_DIRS[${tier}]}"
  num_ranks="${TIER_NUM_RANKS[${tier}]}"
  num_proc="${TIER_NUM_PROC[${tier}]}"

  base_output="${OUTPUT_BASE}/${BASE_TOKENIZER_SLUG}/${tier}"
  base_job="${JOB_PREFIX_BASE}_${tier}_orig"
  submit_pipeline "${shard_dir}" "${BASE_TOKENIZER_NAME}" "${base_output}" "${base_job}" "${num_ranks}" "${num_proc}"

  ext_tokenizer="${TIER_EXT_TOKENIZERS[${tier}]}"
  if [[ ! -d "${ext_tokenizer}" ]]; then
    echo "[ERROR] Extended tokenizer directory ${ext_tokenizer} missing for ${tier}" >&2
    exit 1
  fi
  ext_slug="${TIER_EXT_SLUGS[${tier}]}"
  ext_output="${OUTPUT_BASE}/${ext_slug}"
  ext_job="${JOB_PREFIX_BASE}_${tier}_extended"
  submit_pipeline "${shard_dir}" "${ext_tokenizer}" "${ext_output}" "${ext_job}" "${num_ranks}" "${num_proc}"
done

echo "[INFO] Submitted CoLA tier tokenization jobs for both original and extended tokenizers."
