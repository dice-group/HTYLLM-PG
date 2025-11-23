#!/usr/bin/env bash
# Submit tokenization+merge jobs for all tokenizer/subset combinations defined below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/tokenize_and_merge_pipeline.sh"

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "Missing pipeline script: ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

NUM_RANKS=1
NUM_PROC=1
CPUS_PER_TASK=4
MEM_PER_TASK=32G
TIME_LIMIT=02:00:00
MERGE_CPUS=4
MERGE_MEM=64G
MERGE_TIME=04:00:00
MERGE_WORKERS=""
TRANSFORMERS_OFFLINE=1
EVAL_FRACTION=0.05
EVAL_SEED=42
MERGE_SPLIT_FRACTION=0.05
MERGE_SPLIT_SEED=42

usage() {
  cat <<EOF
Usage: $0 --shard-dir PATH --output-base PATH [options]

Required arguments:
  --shard-dir PATH      Root directory that holds per-language shard folders.
  --output-base PATH    Base directory for outputs (e.g., /scratch/.../hierarchical_adapter).

Options (propagated to tokenize_and_merge_pipeline.sh):
  --num-ranks INT       Number of SLURM array tasks/ranks (default: ${NUM_RANKS}).
  --num-proc INT        --num_proc for tokenize_slurm.py (default: ${NUM_PROC}).
  --cpus-per-task INT   CPUs per tokenization task (default: ${CPUS_PER_TASK}).
  --mem MEM             Memory per tokenization task (default: ${MEM_PER_TASK}).
  --time HH:MM:SS       Time limit per tokenization task (default: ${TIME_LIMIT}).
  --merge-cpus INT      CPUs for merge job (default: ${MERGE_CPUS}).
  --merge-mem MEM       Memory for merge job (default: ${MERGE_MEM}).
  --merge-time HH:MM:SS Time limit for merge job (default: ${MERGE_TIME}).
  --merge-workers INT   --max_workers for merge_tokenized_ranks.py (default: match --merge-cpus).
  --trans-offline {0,1} Set TRANSFORMERS_OFFLINE flag (default: ${TRANSFORMERS_OFFLINE}).
  --eval-fraction FLOAT Validation fraction per language during tokenization (default: ${EVAL_FRACTION}).
  --eval-seed INT       Seed for eval hashing (default: ${EVAL_SEED}).
  --merge-split-fraction FLOAT Validation fraction reserved during merge (default: ${MERGE_SPLIT_FRACTION}).
  --merge-split-seed INT Seed used if merge falls back to random split (default: ${MERGE_SPLIT_SEED}).
  --job-prefix STR      Prefix for SBATCH job names (default derived per combo).
  -h, --help            Show this help message.
EOF
}

SHARD_DIR=""
OUTPUT_BASE=""
JOB_PREFIX_BASE="bulk_tok"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard-dir) SHARD_DIR="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --num-ranks) NUM_RANKS="$2"; shift 2 ;;
    --num-proc) NUM_PROC="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEM_PER_TASK="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --merge-cpus) MERGE_CPUS="$2"; shift 2 ;;
    --merge-mem) MERGE_MEM="$2"; shift 2 ;;
    --merge-time) MERGE_TIME="$2"; shift 2 ;;
    --merge-workers) MERGE_WORKERS="$2"; shift 2 ;;
    --trans-offline) TRANSFORMERS_OFFLINE="$2"; shift 2 ;;
    --job-prefix) JOB_PREFIX_BASE="$2"; shift 2 ;;
    --eval-fraction) EVAL_FRACTION="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --merge-split-fraction) MERGE_SPLIT_FRACTION="$2"; shift 2 ;;
    --merge-split-seed) MERGE_SPLIT_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${SHARD_DIR}" || -z "${OUTPUT_BASE}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

MERGE_WORKERS="${MERGE_WORKERS:-${MERGE_CPUS}}"

#"llama-3.1-8B_tokenizer|meta-llama/Llama-3.1-8B"
TOKENIZERS=(
  "llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B"
  "llama-3.2-3B_tokenizer|meta-llama/Llama-3.2-3B"
)

SUBSETS=(
  "5_langs|five_representatives_mediods"
  "10_langs|ten_representatives_mediods"
  "22_langs|twenty_two_representatives_mediods"
  "46_langs|fourty_six_representatives_mediods"
  "95_langs|ninty_five_representatives_mediods"
  "199_langs|hundred_ninty_nine_representatives_mediods"
  "eng_plus_5_langs|english_plus_five"
  "eng_plus_10_langs|english_plus_ten"
  "eng_plus_22_langs|english_plus_twenty_two"
  "eng_plus_46_langs|english_plus_forty_six"
  "eng_plus_95_langs|english_plus_ninety_five"
  "eng_plus_199_langs|english_plus_hundred_ninety_nine"
)

for tok_entry in "${TOKENIZERS[@]}"; do
  IFS='|' read -r tok_dir tok_name <<<"${tok_entry}"
  for subset_entry in "${SUBSETS[@]}"; do
    IFS='|' read -r subset_dir subset_flag <<<"${subset_entry}"
    output_dir="${OUTPUT_BASE}/${tok_dir}/${subset_dir}"
    job_prefix="${JOB_PREFIX_BASE}_${tok_dir}_${subset_dir}"
    echo "Submitting ${tok_name} / ${subset_flag} -> ${output_dir}"
    "${PIPELINE_SCRIPT}" \
      --shard-dir "${SHARD_DIR}" \
      --tokenizer "${tok_name}" \
      --language-subset "${subset_flag}" \
      --num-ranks "${NUM_RANKS}" \
      --num-proc "${NUM_PROC}" \
      --cpus-per-task "${CPUS_PER_TASK}" \
      --mem "${MEM_PER_TASK}" \
      --time "${TIME_LIMIT}" \
      --merge-cpus "${MERGE_CPUS}" \
      --merge-mem "${MERGE_MEM}" \
      --merge-time "${MERGE_TIME}" \
      --merge-workers "${MERGE_WORKERS}" \
      --eval-fraction "${EVAL_FRACTION}" \
      --eval-seed "${EVAL_SEED}" \
      --merge-split-fraction "${MERGE_SPLIT_FRACTION}" \
      --merge-split-seed "${MERGE_SPLIT_SEED}" \
      --job-prefix "${job_prefix}" \
      --output-root "${output_dir}"
  done
done
