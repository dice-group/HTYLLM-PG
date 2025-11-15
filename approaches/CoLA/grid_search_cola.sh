#!/usr/bin/env bash
# -------------------------------------------------
# grid_search_moe_cola.sh
# -------------------------------------------------

# ---------- SLURM header for the *driver* ----------
#SBATCH --job-name=grid_search_moe_cola
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/driver_%j.log
# -------------------------------------------------

set -euo pipefail

# ----- hyper‑parameter grid -----
LR=5e-5
BATCH_SIZE=32
A_B_PAIRS=(
    "1 2"
    "2 4"
    "3 6"
    "4 8"
    "1 4"
    "2 2"
    "3 4"
    "4 6"
)
COLA_NUM_EXPERTS=4
COLA_TOP_K=1
LORA_RANK=4
LORA_ALPHA=8
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_RATIO=0.06
NUM_TRAIN_EPOCHS=1
SEED=42

for pair in "${A_B_PAIRS[@]}"; do
    read -r NUM_A NUM_B <<< "$pair"

    JOB_NAME="cola_a${NUM_A}_b${NUM_B}"
    LOG_DIR="logs/${JOB_NAME}"
    mkdir -p "$LOG_DIR"

    sbatch \
        --job-name="${JOB_NAME}" \
        --output="${LOG_DIR}/%j.log" \
        --partition=gpu \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=32 \
        --gres=gpu:h100:4 \
        --mem=256G \
        --time=12:00:00 \
        --export=LEARNING_RATE=${LR},PER_DEVICE_TRAIN_BATCH_SIZE=${BATCH_SIZE},SEED=${SEED},\
NUM_A=${NUM_A},NUM_B=${NUM_B},\
COLA_NUM_EXPERTS=${COLA_NUM_EXPERTS},COLA_TOP_K=${COLA_TOP_K},\
LORA_RANK=${LORA_RANK},LORA_ALPHA=${LORA_ALPHA},\
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS},\
WARMUP_RATIO=${WARMUP_RATIO},NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS},\
WANDB_RUN_GROUP="grid_a${NUM_A}_b${NUM_B}" \
        accelerate_moe_cola_train.sh

    echo "[INFO] Submitted ${JOB_NAME}"
done