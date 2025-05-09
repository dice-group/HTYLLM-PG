#!/bin/bash
set -e

DATA_DIR="/data/fineweb2_subset_belebele"
TOKENIZED_DATA_DIR="/data/fineweb2_subset_belebele_tokenized_bloom-560m"
MODEL_NAME="bigscience/bloom-560m"
OUTPUT_DIR="/data/joel/bloom560m-belebele-languages"
LOGGING_DIR="$OUTPUT_DIR/logs"

echo "Running full training pipeline..."
python train_language_adpater.py \
    --data_dir "$DATA_DIR" \
    --tokenized_dir "$TOKENIZED_DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR"
