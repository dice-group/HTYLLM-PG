#!/bin/bash
#SBATCH --job-name=moe-multinode
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1           # 1 DeepSpeed launcher per node
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --partition=normal
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin

set -e

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# ---------- Env ----------
source ~/.bashrc
conda activate moe

echo "Conda environment activated"
echo "Python version: $(python --version)"

export PYTHONUNBUFFERED=1

srun python htyllm_pg/train_tokenizer.py /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset/

echo "Job finished at $(date)"