#!/bin/bash
#SBATCH --job-name=analyze-experts
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

set -e

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# ---------- Env ----------
source ~/.bashrc
conda activate moe

module load system/CUDA/13.0.0
module load compiler/GCCcore/12.3.0

echo "Conda environment activated"
echo "Python version: $(python --version)"

export PYTHONUNBUFFERED=1

# Paths
CHECKPOINT_DIR="/scratch/hpc-prf-merlin/luke/checkpoints_multilingual_3_5b"
CHECKPOINT_TAG="step_124000"
DATA_DIR="/scratch/hpc-prf-merlin/luke/tokenized_multilingual"
OUTPUT_DIR="/scratch/hpc-prf-merlin/luke/expert_analysis"

# Parameters
SAMPLES_PER_LANG=500
BATCH_SIZE=6

echo "Checkpoint: ${CHECKPOINT_DIR}/${CHECKPOINT_TAG}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Samples per language: ${SAMPLES_PER_LANG}"

python -m htyllm_pg.analyze_expert_usage \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --checkpoint-tag ${CHECKPOINT_TAG} \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --samples-per-lang ${SAMPLES_PER_LANG} \
    --batch-size ${BATCH_SIZE}

echo "Job finished at $(date)"
