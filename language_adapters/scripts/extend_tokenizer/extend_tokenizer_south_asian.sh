#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python extend_tokenizer.py \
  --base_tokenizer google/gemma-3-4b-pt \
  --data_dir /upb/users/j/joeldag/profiles/unix/cs/language_adapter_subsets/south_asian \
  --output_dir /upb/users/j/joeldag/profiles/unix/cs/tokenized_data_subsets/gemma_extended_tokenizers/south_asian \
  --added_tokens 3000
