#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=48
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 python embed_flores_langs.py --model-key llama31_8b --batch-size "${BATCH_SIZE}" \
    2>&1 | tee "${LOG_DIR}/llama31_8b.log" &
PID_LLAMA=$!

CUDA_VISIBLE_DEVICES=1 python embed_flores_langs.py --model-key glot500 --batch-size "${BATCH_SIZE}" \
    2>&1 | tee "${LOG_DIR}/glot500.log" &
PID_GLOT=$!

wait "${PID_LLAMA}"
wait "${PID_GLOT}"

echo "Both embedding jobs finished."
