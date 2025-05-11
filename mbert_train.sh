#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Step 1: Running tokenizer.py..."
python src/data/tokenizer.py 

echo "Step 2: Running mBERT_2.py with multi-GPU support..."

# Detect number of GPUs available
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs detected: $NUM_GPUS"

# Use torchrun (preferred for PyTorch >= 1.9)
export TF_CPP_MIN_LOG_LEVEL=3
torchrun --nproc_per_node=$NUM_GPUS src/model/mBERT_2.py 

#python src/model/mBERT_2.py
echo "Step 3: Evaluating the model..."
torchrun --nproc_per_node=$NUM_GPUS src/model/model_eval.py