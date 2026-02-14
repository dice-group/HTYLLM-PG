#!/bin/bash
#SBATCH --job-name=tokenize-data
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1           # 1 process per node
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --mem=64GB
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

INPUT_FOLDER="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples"

OUTPUT_FOLDER="/scratch/hpc-prf-merlin/luke/tokenized_data" 
TOKENIZER_PATH="tokenizer.json"
SEQ_LENGTH=2048
BATCH_SIZE=10000

srun python -m htyllm_pg.tokenize_data ${INPUT_FOLDER} ${OUTPUT_FOLDER} ${TOKENIZER_PATH} ${SEQ_LENGTH} ${BATCH_SIZE}

echo "Job finished at $(date)"

