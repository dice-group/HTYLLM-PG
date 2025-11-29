#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=48

CUDA_VISIBLE_DEVICES=0 python embed_flores_langs.py --model-key llama31_8b --batch-size "$BATCH_SIZE" &
PID_LLAMA=$!

CUDA_VISIBLE_DEVICES=1 python embed_flores_langs.py --model-key glot500 --batch-size "$BATCH_SIZE" &
PID_GLOT=$!

wait "$PID_LLAMA"
wait "$PID_GLOT"

echo "Both embedding jobs finished."
