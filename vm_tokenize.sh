#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run the tokenizer script with correct absolute path
python tokenizer/train_tokenizer.py --files_glob "/data/fineweb2_subset/**/*.jsonl.gz" --output_dir tokenizer

echo "Tokenizer training complete!"
