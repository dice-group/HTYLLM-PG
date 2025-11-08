#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH --mem=120G
#SBATCH --output=logs/tok_simple_%A_%a.out
#SBATCH --error=logs/tok_simple_%A_%a.err
#SBATCH --partition=normal

set -euo pipefail

SHARD_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_shards/"
TOKENIZED_OUTPUT="/scratch/hpc-prf-merlin/joel/tokenized_fw"
TOKENIZER_NAME="/scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizers/256k_vocab/llama-3.2-1B/"
NUM_PROC=4

mkdir -p "$TOKENIZED_OUTPUT" logs

export TRANSFORMERS_OFFLINE=1

srun python -u tokenize_slurm.py \
  --shard_dir "$SHARD_DIR" \
  --save_tokenized_data_dir "$TOKENIZED_OUTPUT" \
  --model_name "$TOKENIZER_NAME" \
  --num_proc "$NUM_PROC"
