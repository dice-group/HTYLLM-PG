#!/usr/bin/env bash
# -------------------------------------------------
# grid_search_moe_cola.sh
# -------------------------------------------------

# -------------------  SLURM HEADER  -------------------
# (kept minimal – the real resources are requested in the
#  inner sbatch calls that launch the training jobs)
#SBATCH --job-name=grid_search_moe_cola
#SBATCH --partition=gpu               # <-- adjust if your cluster uses a different name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # only needed for the driver itself
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/driver_%j.log
# -------------------------------------------------

set -euo pipefail

# ----- hyper‑parameter grids -----
LRs=(5e-5 1e-4 2e-4)
BATCHES=(8 16 32)
SEED=42

for LR in "${LRs[@]}"; do
  for BS in "${BATCHES[@]}"; do
    JOB_NAME="cola_lr${LR}_bs${BS}"
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
        --export=LR=${LR},BATCH_SIZE=${BS},SEED=${SEED},WANDB_RUN_GROUP="grid_lr${LR}_bs${BS}" \
        accelerate_moe_cola_train.sh

      echo "[INFO] Submitted ${JOB_NAME}"
  done
done