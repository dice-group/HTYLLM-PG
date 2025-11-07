#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH --mem=120G
#SBATCH --output=logs/tok_simple_%A_%a.out
#SBATCH --error=logs/tok_simple_%A_%a.err
#SBATCH --partition=normal

set -euo pipefail

DATA_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples/"
OUTPUT_DIR="/scratch/hpc-prf-merlin/joel/joels_test_result_output_dir_for_everything/tokenizing_test"
TOKENIZER_NAME="/scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizers/256k_vocab/llama-3.2-1B/"
MAX_CHUNK_BYTES=$((512 * 1024 * 1024))
NUM_PROC=4

mkdir -p "$OUTPUT_DIR" logs

export TRANSFORMERS_OFFLINE=1

srun python -u tokenize_slurm.py \
  --data_dir "$DATA_DIR" \
  --save_tokenized_data_dir "$OUTPUT_DIR" \
  --model_name "$TOKENIZER_NAME" \
  --num_proc "$NUM_PROC" \
  --max_chunk_bytes "$MAX_CHUNK_BYTES"
