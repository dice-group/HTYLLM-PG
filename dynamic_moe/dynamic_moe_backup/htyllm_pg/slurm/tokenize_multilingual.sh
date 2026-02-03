#!/bin/bash
#SBATCH --job-name=tokenize-multilingual
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=largemem
#SBATCH --mem=256GB
#SBATCH --account=hpc-prf-merlin

set -e

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

source ~/.bashrc
conda activate moe

echo "Conda environment activated"
echo "Python version: $(python --version)"

export PYTHONUNBUFFERED=1

INPUT_FOLDER="/scratch/hpc-prf-merlin/luke/fineweb_samples/sharded_samples"
OUTPUT_FOLDER="/scratch/hpc-prf-merlin/luke/tokenized_multilingual"
TOKENIZER_PATH="tokenizer.json"
SEQ_LENGTH=2048
BATCH_SIZE=10000

srun python -m htyllm_pg.tokenize_data ${INPUT_FOLDER} ${OUTPUT_FOLDER} ${TOKENIZER_PATH} ${SEQ_LENGTH} ${BATCH_SIZE}

echo "Job finished at $(date)"

