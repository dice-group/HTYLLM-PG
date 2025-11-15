#!/bin/bash

# Helper script to queue two CoLA runs:
#   1) Baseline PiSSA init using defaults from accelerate_moe_cola_train.sh
#   2) Non-PiSSA init with NUM_A=NUM_B=1 (still 4 experts, top-k=1)

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT="${SCRIPT_DIR}/accelerate_moe_cola_train.sh"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Could not find accelerate_moe_cola_train.sh at ${TRAIN_SCRIPT}" >&2
  exit 1
fi

echo "[INFO] Submitting PiSSA-enabled CoLA run (defaults)."
PISSA_JOB_ID=$(sbatch --export=ALL,USE_COLA_PISSA_INIT=True,COLA_INIT_LORA_WEIGHTS= "${TRAIN_SCRIPT}" | awk '{print $4}')
echo "[INFO] Submitted PiSSA job ${PISSA_JOB_ID}"

echo "[INFO] Submitting non-PiSSA CoLA run (NUM_A=NUM_B=1)."
NOPISSA_JOB_ID=$(sbatch --export=ALL,USE_COLA_PISSA_INIT=False,COLA_INIT_LORA_WEIGHTS=,NUM_A=1,NUM_B=1 "${TRAIN_SCRIPT}" | awk '{print $4}')
echo "[INFO] Submitted non-PiSSA job ${NOPISSA_JOB_ID}"

echo "[INFO] Both runs queued. Use 'squeue -u $USER' to monitor progress."
