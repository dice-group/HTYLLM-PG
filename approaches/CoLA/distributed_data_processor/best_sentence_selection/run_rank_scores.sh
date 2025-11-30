#!/usr/bin/env bash
# --------------------------------------------------------------
# Distributed joint‑score computation: one job per rank_XXXXX folder
# --------------------------------------------------------------
set -euo pipefail

# ==== USER SETTINGS ==================================================
ROOT_DIR=$1
GLOBAL_COUNTS_DIR="${ROOT_DIR}/global_counts"      # <-- contains word_counts.json & subword_counts.json
TOKENIZER="meta-llama/Llama-3.1-8B"
ALPHA=0.5
BETA=0.5
WINDOW=5
JOB_PREFIX="score"
LOG_ROOT="${ROOT_DIR}/logs"
mkdir -p "${LOG_ROOT}"
# ====================================================================

# Discover how many rank_* directories exist (same logic as merge_tokenized_ranks.py)
NUM_RANKS=$(find "${ROOT_DIR}" -maxdepth 1 -type d -name "rank_*" | wc -l)
echo "Found ${NUM_RANKS} rank directories → launching array job 0-$((NUM_RANKS-1))"

sbatch --parsable \
    --array=0-$((NUM_RANKS-1)) \
    --job-name="${JOB_PREFIX}_rank" \
    --output="${LOG_ROOT}/score_%A_%a.log" \
    --error="${LOG_ROOT}/score_%A_%a.err" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --mem=120G --time=01:30:00 \
    --partition=normal \
    --wrap "\
        IDX=\$SLURM_ARRAY_TASK_ID; \
        RANK_DIR=\$(printf '%s/rank_%05d' \"${ROOT_DIR}\" \"\$IDX\"); \
        python -u rank_sentence_scores_one_rank.py \
            --rank-dir \"\$RANK_DIR\" \
            --global-counts-dir \"${GLOBAL_COUNTS_DIR}\" \
            --tokenizer \"${TOKENIZER}\" \
            --alpha ${ALPHA} --beta ${BETA} --window ${WINDOW} \
    "