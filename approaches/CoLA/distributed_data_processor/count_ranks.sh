#!/usr/bin/env bash
# --------------------------------------------------------------
# count_ranks.sh – launch one job per rank_* folder
# --------------------------------------------------------------
set -euo pipefail

# ---- USER SETTINGS -------------------------------------------------
RANK_ROOT=$1
TOKENIZER="meta-llama/Llama-3.1-8B"
# OUT_ROOT="/scratch/hpc-prf-merlin/yven/tokenized/llama-3.1-8B_tokenizer/eng_plus_5_langs_counts"
LOG_ROOT="${RANK_ROOT}/logs"
mkdir -p "${LOG_ROOT}" "${RANK_ROOT}"

# ---- discover how many rank_* dirs exist ----------------------------
NUM_RANKS=$(find "${RANK_ROOT}" -maxdepth 1 -type d -name "rank_*" | wc -l)
echo "Found ${NUM_RANKS} rank directories → will launch ${NUM_RANKS} array tasks"

# ---- sbatch array -------------------------------------------------
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
        echo \"🟢  Rank \$RANK_IDX → \${RANK_DIR}\"; \
        python -u collect_counts.py \
            --rank-dir \"\$RANK_DIR\" \
            --tokenizer \"${TOKENIZER}\" \
            --out-dir \"\$OUT_DIR\" \
        "