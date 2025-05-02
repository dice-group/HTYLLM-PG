#!/bin/bash

TOTAL_DOCS=1000000
NUM_LANGUAGES=150
OUTPUT_DIR="/data/fineweb2_subset"
META_FILE="fineweb2_meta.json"
LOG_LEVEL="INFO"
INCLUDE_ENGLISH="--include_english"

python sample_fineweb2.py \
  --total_docs $TOTAL_DOCS \
  --num_languages $NUM_LANGUAGES \
  --output_dir $OUTPUT_DIR \
  --meta_file $META_FILE \
  --log_level $LOG_LEVEL \
  $INCLUDE_ENGLISH
