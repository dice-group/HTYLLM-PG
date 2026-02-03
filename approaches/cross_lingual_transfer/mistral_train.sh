#!/bin/bash

# Mistral-7B Fine-tuning Runner Script
# Usage: ./run_mistral_finetune.sh

set -e  # Exit on any error
export CUDA_VISIBLE_DEVICES=0

# Configuration
DATA_PATH="/data/sashreek/tokenized_adapter_subsets/mistral7b/final_model/crosslingual_mistral7b_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/tokenized_data/combined_dataset.bin"
SCRIPT_NAME="src/model/mistral_crosslingual_training.py"
LOG_DIR="./mistral_logs"
LOG_FILE="${LOG_DIR}/mistral_finetune_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Print configuration
echo "============================================="
echo "Mistral-7B Fine-tuning Configuration"
echo "============================================="
echo "Data path: $DATA_PATH"
echo "Script: $SCRIPT_NAME"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "============================================="

# # Check if data file exists
# if [ ! -f "$DATA_PATH" ]; then
#     echo "ERROR: Data file not found at $DATA_PATH"
#     exit 1
# fi

# # Check if Python script exists
# if [ ! -f "$SCRIPT_NAME" ]; then
#     echo "ERROR: Python script $SCRIPT_NAME not found"
#     exit 1
# fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo "============================================="

# Run the fine-tuning script
echo "Starting fine-tuning process..."
echo "Logging to: $LOG_FILE"


# Run with both console output and log file
python "$SCRIPT_NAME" "$DATA_PATH" 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "============================================="
    echo "Fine-tuning completed successfully!"
    echo "End time: $(date)"
    echo "Log saved to: $LOG_FILE"
    echo "============================================="
else
    echo "============================================="
    echo "Fine-tuning failed! Check logs for details."
    echo "Log file: $LOG_FILE"
    echo "============================================="
    exit 1
fi