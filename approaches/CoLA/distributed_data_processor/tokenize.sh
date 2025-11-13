#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=logs/tok_simple_%A_%a.log
#SBATCH --partition=normal

# Input a dir which includes shards in the same size
# use slurm array and disitrbutes tokenization of each of those shards equally
# Takes ~5 min for 90 GB of compressed data with the compute resources above, when equally sharded

set -euo pipefail

PARTS_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples"
TOKENIZED_OUTPUT="/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer/5_langs"
TOKENIZER_NAME="meta-llama/Llama-3.1-8B"
NUM_PROC=4
LANGUAGE_SUBSET="five_representatives_mediods"

mkdir -p "$TOKENIZED_OUTPUT" logs

export TRANSFORMERS_OFFLINE=1

srun python -u tokenize_slurm.py \
  --shard_dir "$PARTS_DIR" \
  --save_tokenized_data_dir "$TOKENIZED_OUTPUT" \
  --model_name "$TOKENIZER_NAME" \
  --num_proc "$NUM_PROC" \
  ${LANGUAGE_SUBSET:+--language_subset "$LANGUAGE_SUBSET"}
