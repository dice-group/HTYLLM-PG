#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --array=0-199
#SBATCH --mem=16G
#SBATCH --output=logs/tok_simple_%A_%a.out
#SBATCH --error=logs/tok_simple_%A_%a.err
#SBATCH --partition=normal

# Input a dir which includes shards in the same size
# use slurm array and disitrbutes tokenization of each of those shards equally
# Takes ~5 min for 90 GB of compressed data with the compute resources above, when equally sharded

set -euo pipefail

PARTS_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples"
TOKENIZED_OUTPUT="/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/200_all_langs/llama_3.2-1B_256k_multilingual_tokenizer"
TOKENIZER_NAME="/scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizers/256k_vocab/llama-3.2-1B/"
NUM_PROC=4

mkdir -p "$TOKENIZED_OUTPUT" logs

export TRANSFORMERS_OFFLINE=1

srun python -u tokenize_slurm.py \
  --shard_dir "$PARTS_DIR" \
  --save_tokenized_data_dir "$TOKENIZED_OUTPUT" \
  --model_name "$TOKENIZER_NAME" \
  --num_proc "$NUM_PROC"
