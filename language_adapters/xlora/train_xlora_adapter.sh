#!/bin/bash

TOKENIZED_DATA_DIR="/data/joel/tokenized_adapter_subsets/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
OUTPUT_DIR="/data/joel/results_language_adapters/xlora/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian"
LOGGING_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGGING_DIR"

export CUDA_VISIBLE_DEVICES=1

#accelerate launch --config_file ../accelerate_config_gpu23.yaml train_xlora_adapter.py \
python train_xlora_adapter.py \
  --tokenized_dir "$TOKENIZED_DATA_DIR" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR"