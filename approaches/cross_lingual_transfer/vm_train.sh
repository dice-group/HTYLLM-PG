#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Step 1: Running tokenizer.py..."
python src/data/tokenizer.py src/data/fineweb2_subset/arb_Arab.jsonl

echo "Step 2: Running gemma training script..."

# Detect number of GPUs available
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs detected: $NUM_GPUS"

# Use torchrun (preferred for PyTorch >= 1.9)
#export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=1
#torchrun --nproc_per_node=$NUM_GPUS src/model/gemma-3-4b.py src/data/fineweb2_subset/tokenized_data/arb_Arab.bin
python src/model/gemma-3-4b.py src/data/fineweb2_subset/tokenized_data/arb_Arab.bin