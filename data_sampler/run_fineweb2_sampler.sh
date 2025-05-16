#!/bin/bash

TOTAL_DOCS=14000000 #20 000 000 ~ 95GB #14 000 000 ~ 67.37gb of data
NUM_LANGUAGES=190
OUTPUT_DIR="/data/fineweb2_subset_190_most_used_languages"
META_FILE="filtered_fineweb2_meta.json"

python sample_fineweb2.py \
  --total_docs $TOTAL_DOCS \
  --num_languages $NUM_LANGUAGES \
  --output_dir $OUTPUT_DIR \
  --meta_file $META_FILE \
