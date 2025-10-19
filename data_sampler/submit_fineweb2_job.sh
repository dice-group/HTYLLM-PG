#!/bin/bash
#SBATCH --job-name=fineweb2
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-merlin

# Activate the conda environment
source ~/miniconda3/bin/activate icebreaker

# Run the Python script with specified arguments
python luke/HTYLLM-PG/sampler/sample_fineweb2.py \
    --total_docs 1621160 \
    --num_languages 18 \
    --output_dir ./data2 \
    --meta_file luke/HTYLLM-PG/sampler/fineweb2_meta.json 