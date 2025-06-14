#!/bin/bash

TOKENIZED_DATA_DIR="/upb/users/j/joeldag/profiles/unix/cs/tokenized_data_subsets/xlora"
MODEL_NAME="bigscience/bloom-560m"
OUTPUT_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/xlora"
LOGGING_DIR="$OUTPUT_DIR/logs"

export CUDA_VISIBLE_DEVICES=1,2,3

#python train_xlora_adapter.py \
accelerate launch --config_file ../accelerate_config_gpu23.yaml train_xlora_adapter.py \
  --tokenized_dir "$TOKENIZED_DATA_DIR" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR"
