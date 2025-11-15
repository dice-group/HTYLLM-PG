#!/bin/bash

# Helper script to queue two CoLA runs:
#   1) Baseline PiSSA init using defaults from accelerate_moe_cola_train.sh
#   2) Non-PiSSA init with NUM_A=NUM_B=1 (still 4 experts, top-k=1)

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT="${SCRIPT_DIR}/accelerate_moe_cola_train.sh"

# PiSSA configuration
export USE_COLA_PISSA_INIT=True
export COLA_INIT_LORA_WEIGHTS=
export NUM_A=2
export NUM_B=4

echo "[INFO] Submitting PiSSA-enabled CoLA run (defaults)."
PISSA_JOB_ID=$(
  sbatch \
    --job-name=cola-moe-pissa \
    --output=logs/train_acc_pissa_%j.log \
    "${TRAIN_SCRIPT}" \
    | awk '{print $4}'
)
echo "[INFO] Submitted PiSSA job ${PISSA_JOB_ID}"

# non-PiSSA configuration
export USE_COLA_PISSA_INIT=False
export COLA_INIT_LORA_WEIGHTS=
export NUM_A=1
export NUM_B=1

echo "[INFO] Submitting non-PiSSA CoLA run (NUM_A=NUM_B=1)."
NOPISSA_JOB_ID=$(
  sbatch \
    --job-name=cola-moe-nopissa \
    --output=logs/train_acc_nopissa_%j.log \
    "${TRAIN_SCRIPT}" \
    | awk '{print $4}'
)
echo "[INFO] Submitted non-PiSSA job ${NOPISSA_JOB_ID}"

echo "[INFO] Both runs queued. Use 'squeue -u $USER' to monitor progress."
