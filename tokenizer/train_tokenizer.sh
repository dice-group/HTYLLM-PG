#!/bin/bash

#SBATCH --job-name=train-tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --mem=128GB
#SBATCH --account=hpc-prf-merlin

# Activate the conda environment
source ~/miniconda3/bin/activate icebreaker

# Run tokenization on a single node
srun --nodes=1 --ntasks=1 python train_tokenizer.py

echo "Tokenization complete!" 