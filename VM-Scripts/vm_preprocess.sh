#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")/.."

# Run preprocessing on a single node with the correct absolute path
python src/preprocess.py --files "/data/fineweb2_subset/**/*.jsonl.gz" --tokenizer tokenizer --num_proc 32 --out_dir data/processed

echo "Preprocessing complete!"
