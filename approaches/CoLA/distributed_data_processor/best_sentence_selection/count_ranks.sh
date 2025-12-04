#!/usr/bin/env bash
# Count tokens for every rank_* directory.
set -euo pipefail

usage() {
  echo "Usage: $0 RANK_ROOT [TOKENIZER]" >&2
  echo "Example: $0 /scratch/.../5_langs_ranks meta-llama/Llama-3.1-8B" >&2
  exit 1
}

[[ $# -lt 1 || $# -gt 2 ]] && usage

RANK_ROOT="$1"
TOKENIZER="${2:-meta-llama/Llama-3.1-8B}"
LOG_ROOT="${RANK_ROOT}/logs"
mkdir -p "${LOG_ROOT}" "${RANK_ROOT}"

NUM_RANKS=$(find "${RANK_ROOT}" -maxdepth 1 -type d -name "rank_*" | wc -l)
echo "[INFO] Found ${NUM_RANKS} rank directories under ${RANK_ROOT}"

sbatch --parsable \
    --array=0-$((NUM_RANKS-1)) \
    --job-name=cnt_rank \
    --output="${LOG_ROOT}/cnt_%A_%a.log" \
    --error="${LOG_ROOT}/cnt_%A_%a.err" \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --partition=normal \
    --wrap "\
        RANK_IDX=\$SLURM_ARRAY_TASK_ID; \
        RANK_DIR=\$(printf '%s/rank_%05d' \"${RANK_ROOT}\" \"\$RANK_IDX\"); \
        OUT_DIR=\$(printf '%s/rank_%05d' \"${RANK_ROOT}\" \"\$RANK_IDX\"); \
        echo '[INFO] Counting' \"\$RANK_DIR\"; \
        python -u collect_counts.py \
            --rank-dir \"\$RANK_DIR\" \
            --tokenizer \"${TOKENIZER}\" \
            --out-dir \"\$OUT_DIR\" \
        "
