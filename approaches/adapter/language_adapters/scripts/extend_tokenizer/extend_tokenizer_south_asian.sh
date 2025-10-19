#!/bin/bash

python extend_tokenizer.py \
  --base_tokenizer google/gemma-3-4b-pt \
  --data_dir /data/joel/language_adapters_subsets/south_asian \
  --output_dir /data/joel/tokenized_adapter_subsets/gemma_extended_tokenizers/south_asian \
  --added_tokens 3000
