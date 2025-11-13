#!/bin/bash
#SBATCH --job-name=tokenize-subsets
#SBATCH --array=0-5
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

set -e

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"

# ---------- Env ----------
source ~/.bashrc
conda activate moe

echo "Conda environment activated"
echo "Python version: $(python --version)"

export PYTHONUNBUFFERED=1

INPUT_FOLDER="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples"
OUTPUT_FOLDER="/scratch/hpc-prf-merlin/luke/tokenized_subsets"
TOKENIZER_PATH="tokenizer.json"
BATCH_SIZE=10000

# Array of all subsets
SUBSETS=(
    "five_representatives_mediods"
    "ten_representatives_mediods"
    "twenty_two_representatives_mediods"
    "fourty_six_representatives_mediods"
    "ninty_five_representatives_mediods"
    "hundred_ninty_nine_representatives_mediods"
)

SUBSET_NAME=${SUBSETS[$SLURM_ARRAY_TASK_ID]}
echo "Processing subset: ${SUBSET_NAME}"

srun python -m htyllm_pg.tokenize_subsets ${INPUT_FOLDER} ${OUTPUT_FOLDER}/${SUBSET_NAME} ${TOKENIZER_PATH} ${SUBSET_NAME} ${BATCH_SIZE}

echo "Job finished at $(date)"

