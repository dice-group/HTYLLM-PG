#!/usr/bin/env bash
# Tokenize a sharded corpus with a SLURM array and merge rank_* datasets afterward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_RANKS=1
NUM_PROC=1
CPUS_PER_TASK=4
MEM_PER_TASK=32G
TIME_LIMIT=02:00:00
PARTITION=normal
MERGE_CPUS=4
MERGE_MEM=64G
MERGE_TIME=04:00:00
JOB_PREFIX="tok"
TRANSFORMERS_OFFLINE=1

usage() {
  cat <<EOF
Usage: $0 --shard-dir PATH --tokenizer NAME --output-root PATH [options]

Required arguments:
  --shard-dir PATH          Root directory that holds the per-language shard folders.
  --tokenizer NAME          HF tokenizer name or path (passed to tokenize_slurm.py).
  --output-root PATH        Final dataset location (e.g., /.../llama-3.2-1B_tokenizer/5_langs).

Common options:
  --language-subset NAME    Name defined in language_subsets.py (e.g., five_representatives_mediods).
  --languages "A B C"       Explicit language folder list (mutually exclusive with --language-subset).
  --num-ranks INT           Number of SLURM array tasks/ranks to launch (default: ${NUM_RANKS}).
  --num-proc INT            Value for --num_proc passed to tokenize_slurm.py (default: ${NUM_PROC}).
  --cpus-per-task INT       CPUs per SLURM task for tokenization (default: ${CPUS_PER_TASK}).
  --mem MEM                 Memory per task for tokenization (default: ${MEM_PER_TASK}).
  --time HH:MM:SS           Time limit per tokenization task (default: ${TIME_LIMIT}).
  --partition NAME          SLURM partition to use (default: ${PARTITION}).
  --merge-cpus INT          CPUs for merge job (default: ${MERGE_CPUS}).
  --merge-mem MEM           Memory for merge job (default: ${MERGE_MEM}).
  --merge-time HH:MM:SS     Time limit for merge job (default: ${MERGE_TIME}).
  --merge-workers INT       --max_workers passed to merge_tokenized_ranks.py (default: ${MERGE_CPUS}).
  --job-prefix STR          Prefix for SLURM job names (default: ${JOB_PREFIX}).
  --log-root PATH           Directory to store log files (default: <output-root>_logs).
  --trans-offline {0,1}     Set TRANSFORMERS_OFFLINE env flag (default: ${TRANSFORMERS_OFFLINE}).

Example:
  $0 --shard-dir /scratch/.../sharded_samples \\
     --tokenizer meta-llama/Llama-3.2-1B \\
     --language-subset five_representatives_mediods \\
     --num-ranks 64 \\
     --num-proc 4 \\
     --output-root /scratch/.../hierarchical_adapter/llama-3.2-1B_tokenizer/5_langs
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
    --partition) PARTITION="$2"; shift 2 ;;
    --merge-cpus) MERGE_CPUS="$2"; shift 2 ;;
    --merge-mem) MERGE_MEM="$2"; shift 2 ;;
    --merge-time) MERGE_TIME="$2"; shift 2 ;;
    --merge-workers) MERGE_WORKERS="$2"; shift 2 ;;
    --job-prefix) JOB_PREFIX="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --trans-offline) TRANSFORMERS_OFFLINE="$2"; shift 2 ;;
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

if [[ "${NUM_RANKS}" -lt 1 ]]; then
  echo "--num-ranks must be >= 1" >&2
  exit 1
fi

MERGE_WORKERS="${MERGE_WORKERS:-${MERGE_CPUS}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}_logs}"
RANK_OUTPUT_DIR="${OUTPUT_ROOT}_ranks"
mkdir -p "${LOG_ROOT}" "${RANK_OUTPUT_DIR}"

printf -v TOKEN_CMD 'python -u %q/tokenize_slurm.py --shard_dir %q --save_tokenized_data_dir %q --model_name %q --num_proc %q' \
  "${SCRIPT_DIR}" "${SHARD_DIR}" "${RANK_OUTPUT_DIR}" "${TOKENIZER}" "${NUM_PROC}"

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
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --partition="${PARTITION}" \
    --export=ALL,TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    --output="${LOG_ROOT}/tokenize_%A_%a.out" \
    --error="${LOG_ROOT}/tokenize_%A_%a.err" \
    --wrap "${TOKEN_CMD}"
)

printf -v MERGE_CMD 'python -u %q/merge_tokenized_ranks.py --tokenized_root %q --output_path %q --overwrite --workers %q' \
  "${SCRIPT_DIR}" "${RANK_OUTPUT_DIR}" "${OUTPUT_ROOT}" "${MERGE_WORKERS}"

MERGE_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:${TOKEN_JOB_ID} \
    --job-name="${JOB_PREFIX}_merge" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${MERGE_CPUS}" \
    --mem="${MERGE_MEM}" \
    --time="${MERGE_TIME}" \
    --partition="${PARTITION}" \
    --export=ALL,TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    --output="${LOG_ROOT}/merge_%j.out" \
    --error="${LOG_ROOT}/merge_%j.err" \
    --wrap "${MERGE_CMD}"
)

cat <<EOF
Submitted tokenization array job: ${TOKEN_JOB_ID}
Submitted merge job (after tokenization): ${MERGE_JOB_ID}
Rank outputs: ${RANK_OUTPUT_DIR}
Final dataset will be saved to: ${OUTPUT_ROOT}
Logs stored in: ${LOG_ROOT}
EOF
