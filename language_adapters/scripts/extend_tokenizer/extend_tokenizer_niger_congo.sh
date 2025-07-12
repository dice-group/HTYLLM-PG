#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python extend_tokenizer.py \
  --base_tokenizer mistralai/Mistral-7B-v0.3 \
  --data_dir /data/adapter_fineweb2_subset/niger_congo/ \
  --output_dir /data/joel/extended_tokenizers/mistral7b/niger_congo/ \
  --added_tokens 3000
