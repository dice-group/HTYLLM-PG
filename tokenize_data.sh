#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

# Activate the conda environment
source ~/miniconda3/bin/activate icebreaker

# Run tokenization on a single node
srun --nodes=1 --ntasks=1 python src/data/fineweb.py

echo "Tokenization complete!" 