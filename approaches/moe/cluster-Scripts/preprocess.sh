#!/bin/bash

#SBATCH --job-name=preprocess-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

# Activate the conda environment
source ~/miniconda3/bin/activate icebreaker

# Run preprocessing on a single node
srun --nodes=1 --ntasks=1 python src/preprocess.py --files "../../data/**/*.jsonl.gz" --tokenizer tokenizer --num_proc 32

echo "Preprocessing complete!" 