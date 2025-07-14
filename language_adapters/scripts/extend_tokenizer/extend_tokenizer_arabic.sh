#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python extend_tokenizer.py \
  --base_tokenizer mistralai/Mistral-7B-v0.3 \
  --data_dir  /data/adapter_fineweb2_subset/arabic/ \
  --output_dir /data/joel/extended_tokenizers/mistral7b/arabic/ \
  --added_tokens 8000
