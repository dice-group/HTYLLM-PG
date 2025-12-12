#!/bin/bash
#SBATCH --job-name=python_split_fast
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=172
#SBATCH --mem=430G
#SBATCH --time=01:00:00
#SBATCH --output=logs/python_split-%j.log

# This takes all large files (>512 MB)
# and separates them for efficient tokenization.
# Takes ~10 min for 90 GB of compressed data with the compute resources above
# i had problems using this one time with slurm, then just run it on normal logic node

echo "Running Python-based parallel split with timing..."

INPUT_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_12_tier_samples/samples"
OUTPUT_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_12_tier_samples/sharded_samples"

time python -u shard_python.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR"