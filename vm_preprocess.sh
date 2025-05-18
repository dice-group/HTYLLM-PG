#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run preprocessing on a single node
# Adjust the path to your data files as needed
python src/preprocess.py --files "data/**/*.jsonl.gz" --tokenizer tokenizer --num_proc 32 --out_dir data/processed

echo "Preprocessing complete!"
