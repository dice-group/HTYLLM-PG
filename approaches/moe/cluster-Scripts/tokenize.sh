#!/bin/bash

#SBATCH --job-name=tokenize-data-moe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --mem=128GB
#SBATCH --qos=express
#SBATCH --account=hpc-prf-merlin

# Activate the conda environment
source ~/miniconda3/bin/activate icebreaker 
echo "Tokenizing data..."
# Run tokenization on a single node
srun --nodes=1 --ntasks=1 python tokenizer/train_tokenizer.py --files_glob "/scratch/hpc-prf-merlin/htyllm-pg/luke/data/**/*.jsonl.gz" --output_dir tokenizer --vocab_size 50304

echo "Tokenization complete!"