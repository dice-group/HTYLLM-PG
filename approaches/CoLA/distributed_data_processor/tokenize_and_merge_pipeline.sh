#!/usr/bin/env bash
# Tokenize a sharded corpus with a SLURM array and merge rank_* datasets afterward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_RANKS=1
NUM_PROC=1
CPUS_PER_TASK=4
MEM_PER_TASK=32G
TIME_LIMIT=02:00:00
MERGE_CPUS=4
MERGE_MEM=64G
MERGE_TIME=04:00:00
JOB_PREFIX="tok"
EVAL_FRACTION=0.05
EVAL_SEED=42
MERGE_SPLIT_FRACTION=0.05
MERGE_SPLIT_SEED=42

usage() {
  cat <<EOF
Usage: $0 --shard-dir PATH --tokenizer NAME --output-root PATH [options]

Key options:
  --language-subset NAME    Use a subset from language_subsets.py.
  --languages "L1 L2"       Explicit language directories (mutually exclusive).
  --num-ranks INT           SLURM array size (overrides subset defaults below).
  --num-proc INT            Workers passed to tokenize_slurm.py --num_proc.
  --cpus-per-task INT       CPUs per tokenization task.
  --merge-workers INT       Worker count for merge_tokenized_ranks.py.
  --eval-fraction FLOAT     Fraction per language for validation tagging during tokenization (default: ${EVAL_FRACTION}).
  --eval-seed INT           Seed for eval hashing (default: ${EVAL_SEED}).
  --merge-split-fraction FLOAT  Fraction to reserve for validation when saving merged dataset (default: ${MERGE_SPLIT_FRACTION}).
  --merge-split-seed INT    Seed used if merge performs fallback split (default: ${MERGE_SPLIT_SEED}).
EOF
}

SHARD_DIR=""
TOKENIZER=""
OUTPUT_ROOT=""
LANGUAGE_SUBSET=""
LANGUAGES=""
LOG_ROOT=""
MERGE_WORKERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard-dir) SHARD_DIR="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --language-subset) LANGUAGE_SUBSET="$2"; shift 2 ;;
    --languages) LANGUAGES="$2"; shift 2 ;;
    --num-ranks) NUM_RANKS="$2"; shift 2 ;;
    --num-proc) NUM_PROC="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEM_PER_TASK="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --merge-cpus) MERGE_CPUS="$2"; shift 2 ;;
    --merge-mem) MERGE_MEM="$2"; shift 2 ;;
    --merge-time) MERGE_TIME="$2"; shift 2 ;;
    --merge-workers) MERGE_WORKERS="$2"; shift 2 ;;
    --job-prefix) JOB_PREFIX="$2"; shift 2 ;;
    --eval-fraction) EVAL_FRACTION="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --merge-split-fraction) MERGE_SPLIT_FRACTION="$2"; shift 2 ;;
    --merge-split-seed) MERGE_SPLIT_SEED="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${SHARD_DIR}" || -z "${TOKENIZER}" || -z "${OUTPUT_ROOT}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ -n "${LANGUAGE_SUBSET}" && -n "${LANGUAGES}" ]]; then
  echo "Use either --language-subset or --languages, not both." >&2
  exit 1
fi

if [[ -n "${LANGUAGE_SUBSET}" ]]; then
  preset_ranks=""
  preset_merge_mem=""
  preset_merge_time=""
  case "${LANGUAGE_SUBSET}" in
    five_representatives_mediods|english_plus_five)
      preset_ranks=5
      preset_merge_mem=64G
      preset_merge_time=02:00:00
      ;;
    ten_representatives_mediods|english_plus_ten)
      preset_ranks=10
      preset_merge_mem=80G
      preset_merge_time=03:00:00
      ;;
    twenty_two_representatives_mediods|english_plus_twenty_two)
      preset_ranks=22
      preset_merge_mem=96G
      preset_merge_time=04:00:00
      ;;
    fourty_six_representatives_mediods|english_plus_forty_six)
      preset_ranks=46
      preset_merge_mem=128G
      preset_merge_time=06:00:00
      ;;
    ninty_five_representatives_mediods|english_plus_ninety_five)
      preset_ranks=95
      preset_merge_mem=160G
      preset_merge_time=08:00:00
      ;;
    hundred_ninty_nine_representatives_mediods|english_plus_hundred_ninety_nine)
      preset_ranks=100
      preset_merge_mem=192G
      preset_merge_time=10:00:00
      ;;
  esac
  if [[ -n "${preset_ranks}" ]]; then
    NUM_RANKS="${preset_ranks}"
    echo "Using ${NUM_RANKS} SLURM ranks for subset ${LANGUAGE_SUBSET}."
  fi
  if [[ -n "${preset_merge_mem}" ]]; then
    MERGE_MEM="${preset_merge_mem}"
  fi
  if [[ -n "${preset_merge_time}" ]]; then
    MERGE_TIME="${preset_merge_time}"
  fi
fi

if [[ "${NUM_RANKS}" -lt 1 ]]; then
  echo "--num-ranks must be >= 1" >&2
  exit 1
fi

MERGE_WORKERS="${MERGE_WORKERS:-${MERGE_CPUS}}"
LOG_ROOT="${LOG_ROOT:-logs/${JOB_PREFIX}}"
RANK_OUTPUT_DIR="${OUTPUT_ROOT}_ranks"
mkdir -p "${LOG_ROOT}" "${RANK_OUTPUT_DIR}"

printf -v TOKEN_CMD 'python -u %q/tokenize_slurm.py --shard_dir %q --save_tokenized_data_dir %q --model_name %q --num_proc %q --eval_fraction %q --eval_seed %q' \
  "${SCRIPT_DIR}" "${SHARD_DIR}" "${RANK_OUTPUT_DIR}" "${TOKENIZER}" "${NUM_PROC}" "${EVAL_FRACTION}" "${EVAL_SEED}"

if [[ -n "${LANGUAGE_SUBSET}" ]]; then
  printf -v TOKEN_CMD '%s --language_subset %q' "${TOKEN_CMD}" "${LANGUAGE_SUBSET}"
elif [[ -n "${LANGUAGES}" ]]; then
  TOKEN_CMD+=" --languages ${LANGUAGES}"
fi

TOKEN_JOB_ID=$(
  sbatch --parsable \
    --array=0-$((NUM_RANKS - 1)) \
    --job-name="${JOB_PREFIX}_tok" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --partition=normal \
    --export=ALL \
    --output="${LOG_ROOT}/tokenize_%A_%a.log" \
    --wrap "${TOKEN_CMD}"
)

printf -v MERGE_CMD 'python -u %q/merge_tokenized_ranks.py --tokenized_root %q --output_path %q --overwrite --workers %q --split_fraction %q --split_seed %q' \
  "${SCRIPT_DIR}" "${RANK_OUTPUT_DIR}" "${OUTPUT_ROOT}" "${MERGE_WORKERS}" "${MERGE_SPLIT_FRACTION}" "${MERGE_SPLIT_SEED}"

MERGE_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:${TOKEN_JOB_ID} \
    --job-name="${JOB_PREFIX}_merge" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${MERGE_CPUS}" \
    --mem="${MERGE_MEM}" \
    --time="${MERGE_TIME}" \
    --partition=normal \
    --export=ALL \
    --output="${LOG_ROOT}/merge_%j.log" \
    --wrap "${MERGE_CMD}"
)

cat <<EOF
Submitted tokenization array job: ${TOKEN_JOB_ID}
Submitted merge job (after tokenization): ${MERGE_JOB_ID}
Rank outputs: ${RANK_OUTPUT_DIR}
Final dataset will be saved to: ${OUTPUT_ROOT}
Logs stored in: ${LOG_ROOT}
EOF
