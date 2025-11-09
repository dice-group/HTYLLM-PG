#!/bin/bash
#SBATCH --job-name=moe-multinode
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1           # 1 DeepSpeed launcher per node
#SBATCH --cpus-per-task=4
#SBATCH --time=05:30:00
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin

set -e

# ---------- Env ----------
source ~/.bashrc
conda activate moe

python htyllm_pg/train_tokenizer.py /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset/
