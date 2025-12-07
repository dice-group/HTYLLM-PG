#!/bin/bash
#SBATCH --job-name=tokenizer_extension
#SBATCH --output=logs/tokenize_extension/tokenizer_extension_%j.out
#SBATCH --error=logs/tokenize_extension/tokenizer_extension_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=420G
#SBATCH -p normal

srun python -m tokenizer_extension.pipeline --config /scratch/hpc-prf-merlin/joel/moe-study/data_prep/tokenizer_extension/configs/llama3.2-1b.yaml