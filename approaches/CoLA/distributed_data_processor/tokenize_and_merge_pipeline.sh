#!/usr/bin/env bash
# Tokenize a sharded corpus with a SLURM array and merge rank_* datasets afterward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BSS_DIR="${SCRIPT_DIR}/best_sentence_selection"

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
RANKING_ENABLED=0
RANK_ALPHA=0.5
RANK_BETA=0.5
RANK_WINDOW=5
COUNT_CPUS=4
COUNT_MEM=32G
COUNT_TIME=02:00:00
RANK_CPUS=8
RANK_MEM=120G
RANK_TIME=01:30:00
TOPK_SIZES=""
TOPK_MIN_LANGUAGE_SIZE=30000
TOPK_CPUS=8
TOPK_MEM=64G
TOPK_TIME=02:00:00

usage() {
  cat <<EOF
Usage: $0 --shard-dir PATH --tokenizer NAME --output-root PATH [options]

Key options:
  --language-subset NAME    Use a subset from language_subsets.py.
  --languages "L1 L2"       Explicit language directories (mutually exclusive).
  --num-ranks INT           SLURM array size (overrides subset defaults below).
  --num-proc INT            Workers passed to tokenize_slurm.py --num_proc.
  --cpus-per-task INT       CPUs per tokenization task.
  --eval-fraction FLOAT     Fraction per language for validation tagging during tokenization (default: ${EVAL_FRACTION}).
  --eval-seed INT           Seed for eval hashing (default: ${EVAL_SEED}).
  --enable-ranking          Run word/subword counting + joint-score computation before merge.
  --rank-alpha FLOAT        Weight for local popularity Rl (default: ${RANK_ALPHA}).
  --rank-beta FLOAT         Weight for global importance Rg (default: ${RANK_BETA}).
  --rank-window INT         Co-occurrence window size (default: ${RANK_WINDOW}).
  --count-cpus INT          CPUs per count job (default: ${COUNT_CPUS}).
  --count-mem STR           Memory per count job (default: ${COUNT_MEM}).
  --count-time HH:MM:SS     Time limit per count job (default: ${COUNT_TIME}).
  --rank-cpus INT           CPUs per scoring job (default: ${RANK_CPUS}).
  --rank-mem STR            Memory per scoring job (default: ${RANK_MEM}).
  --rank-time HH:MM:SS      Time limit per scoring job (default: ${RANK_TIME}).
  --topk-sizes "10k ..."    Space-separated list of top-K sizes (enables ranking automatically).
  --topk-min-language-size INT  Minimum train examples per language to be considered for top-K datasets (default: ${TOPK_MIN_LANGUAGE_SIZE}).
  --topk-cpus INT           CPUs for top-K filtering job (default: ${TOPK_CPUS}).
  --topk-mem STR            Memory for top-K filtering job (default: ${TOPK_MEM}).
  --topk-time HH:MM:SS      Time limit for top-K filtering job (default: ${TOPK_TIME}).
EOF
}

SHARD_DIR=""
TOKENIZER=""
OUTPUT_ROOT=""
LANGUAGE_SUBSET=""
LANGUAGES=""
LOG_ROOT=""

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
    --job-prefix) JOB_PREFIX="$2"; shift 2 ;;
    --eval-fraction) EVAL_FRACTION="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --enable-ranking) RANKING_ENABLED=1; shift ;;
    --rank-alpha) RANK_ALPHA="$2"; shift 2 ;;
    --rank-beta) RANK_BETA="$2"; shift 2 ;;
    --rank-window) RANK_WINDOW="$2"; shift 2 ;;
    --count-cpus) COUNT_CPUS="$2"; shift 2 ;;
    --count-mem) COUNT_MEM="$2"; shift 2 ;;
    --count-time) COUNT_TIME="$2"; shift 2 ;;
    --rank-cpus) RANK_CPUS="$2"; shift 2 ;;
    --rank-mem) RANK_MEM="$2"; shift 2 ;;
    --rank-time) RANK_TIME="$2"; shift 2 ;;
    --topk-sizes) TOPK_SIZES="$2"; RANKING_ENABLED=1; shift 2 ;;
    --topk-min-language-size) TOPK_MIN_LANGUAGE_SIZE="$2"; shift 2 ;;
    --topk-cpus) TOPK_CPUS="$2"; shift 2 ;;
    --topk-mem) TOPK_MEM="$2"; shift 2 ;;
    --topk-time) TOPK_TIME="$2"; shift 2 ;;
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

TOPK_SIZE_ARRAY=()
if [[ -n "${TOPK_SIZES}" ]]; then
  read -r -a TOPK_SIZE_ARRAY <<< "${TOPK_SIZES}"
  RANKING_ENABLED=1
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
    ten_representatives_mediods|english_plus_ten|lang2vec_auto_best_languages)
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

if [[ "${RANKING_ENABLED}" -eq 1 ]]; then
  TOKEN_CMD+=" --keep_text"
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
    --partition=normal \
    --export=ALL \
    --output="${LOG_ROOT}/tokenize_%A_%a.log" \
    --wrap "${TOKEN_CMD}"
)

MERGE_DEPENDENCY="${TOKEN_JOB_ID}"
GLOBAL_COUNTS_DIR="${RANK_OUTPUT_DIR}/global_counts"
COUNT_JOB_ID=""
COUNT_MERGE_JOB_ID=""
RANK_JOB_ID=""

if [[ "${RANKING_ENABLED}" -eq 1 ]]; then
  COUNT_JOB_ID=$(
    sbatch --parsable \
      --dependency=afterok:${TOKEN_JOB_ID} \
      --array=0-$((NUM_RANKS - 1)) \
      --job-name="${JOB_PREFIX}_cnt" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${COUNT_CPUS}" \
      --mem="${COUNT_MEM}" \
      --time="${COUNT_TIME}" \
      --partition=normal \
      --export=ALL \
      --output="${LOG_ROOT}/count_%A_%a.log" \
      --wrap "\
        IDX=\$SLURM_ARRAY_TASK_ID; \
        RANK_DIR=\$(printf '%s/rank_%05d' \"${RANK_OUTPUT_DIR}\" \"\$IDX\"); \
        python -u ${BSS_DIR}/collect_counts.py \
          --rank-dir \"\$RANK_DIR\" \
          --tokenizer \"${TOKENIZER}\" \
          --out-dir \"\$RANK_DIR\" \
      "
  )

  COUNT_MERGE_JOB_ID=$(
    sbatch --parsable \
      --dependency=afterok:${COUNT_JOB_ID} \
      --job-name="${JOB_PREFIX}_cntmerge" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=2 \
      --mem=32G \
      --time=01:00:00 \
      --partition=normal \
      --export=ALL \
      --output="${LOG_ROOT}/count_merge_%j.log" \
      --wrap "bash ${BSS_DIR}/merge_ranks.sh ${RANK_OUTPUT_DIR}"
  )

  RANK_JOB_ID=$(
    sbatch --parsable \
      --dependency=afterok:${COUNT_MERGE_JOB_ID} \
      --array=0-$((NUM_RANKS - 1)) \
      --job-name="${JOB_PREFIX}_score" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${RANK_CPUS}" \
      --mem="${RANK_MEM}" \
      --time="${RANK_TIME}" \
      --partition=normal \
      --export=ALL \
      --output="${LOG_ROOT}/score_%A_%a.log" \
      --wrap "\
        IDX=\$SLURM_ARRAY_TASK_ID; \
        RANK_DIR=\$(printf '%s/rank_%05d' \"${RANK_OUTPUT_DIR}\" \"\$IDX\"); \
        python -u ${BSS_DIR}/rank_sentence_scores_one_rank.py \
          --rank-dir \"\$RANK_DIR\" \
          --global-counts-dir \"${GLOBAL_COUNTS_DIR}\" \
          --tokenizer \"${TOKENIZER}\" \
          --alpha ${RANK_ALPHA} \
          --beta ${RANK_BETA} \
          --window ${RANK_WINDOW} \
      "
  )
  MERGE_DEPENDENCY="${RANK_JOB_ID}"
fi

printf -v MERGE_CMD 'python -u %q/merge_tokenized_ranks.py --tokenized_root %q --output_path %q --overwrite' \
  "${SCRIPT_DIR}" "${RANK_OUTPUT_DIR}" "${OUTPUT_ROOT}"

MERGE_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:${MERGE_DEPENDENCY} \
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

TOPK_JOB_ID=""
if [[ ${#TOPK_SIZE_ARRAY[@]} -gt 0 ]]; then
  printf -v TOPK_CMD 'python -u %q/create_topk_ranked_datasets.py --merged-root %q --output-prefix %q --min-language-size %q --sizes' \
    "${BSS_DIR}" "${OUTPUT_ROOT}" "${OUTPUT_ROOT}" "${TOPK_MIN_LANGUAGE_SIZE}"
  for size in "${TOPK_SIZE_ARRAY[@]}"; do
    TOPK_CMD+=" ${size}"
  done
  TOPK_JOB_ID=$(
    sbatch --parsable \
      --dependency=afterok:${MERGE_JOB_ID} \
      --job-name="${JOB_PREFIX}_topk" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${TOPK_CPUS}" \
      --mem="${TOPK_MEM}" \
      --time="${TOPK_TIME}" \
      --partition=normal \
      --export=ALL \
      --output="${LOG_ROOT}/topk_%j.log" \
      --wrap "${TOPK_CMD}"
  )
fi

echo "Submitted tokenization array job : ${TOKEN_JOB_ID}"
if [[ -n "${COUNT_JOB_ID}" ]]; then
  echo "Word/subword count job     : ${COUNT_JOB_ID}"
fi
if [[ -n "${COUNT_MERGE_JOB_ID}" ]]; then
  echo "Global count reduction job : ${COUNT_MERGE_JOB_ID}"
fi
if [[ -n "${RANK_JOB_ID}" ]]; then
  echo "Joint-score array job      : ${RANK_JOB_ID}"
fi
echo "Submitted merge job (final HF)   : ${MERGE_JOB_ID}"
if [[ -n "${TOPK_JOB_ID}" ]]; then
  echo "Top-K dataset job          : ${TOPK_JOB_ID}"
fi
echo "Rank outputs directory          : ${RANK_OUTPUT_DIR}"
echo "Final dataset will be saved to  : ${OUTPUT_ROOT}"
echo "Logs stored in                  : ${LOG_ROOT}"
