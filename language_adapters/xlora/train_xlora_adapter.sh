#!/bin/bash

TOKENIZED_DATA_DIR="/data/fineweb2_subset_belebele_tokenized_bloom-560m"
MODEL_NAME="bigscience/bloom-560m"
OUTPUT_DIR="/data/joel/xlora-bloom-560"
LOGGING_DIR="$OUTPUT_DIR/logs"

python train_xlora_adapter.py \
    --tokenized_dir "$TOKENIZED_DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR"
