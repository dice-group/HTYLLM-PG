#!/bin/bash
#SBATCH --job-name=sample-fineweb
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
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

# Set paths
REPO_ROOT=$(pwd)
SAMPLING_SCRIPTS="${REPO_ROOT}/htyllm_pg/sampling"
OUTPUT_ROOT="/scratch/hpc-prf-merlin/luke/fineweb_samples"
INVENTORY_FILE="${REPO_ROOT}/dataset_inventory.json"
QUOTAS_FILE="${REPO_ROOT}/sampling_quotas.csv"
TOKENIZER_DATA="${OUTPUT_ROOT}/tokenizer_training_data.jsonl"
SAMPLED_DATA_DIR="${OUTPUT_ROOT}/sharded_samples"

# Parameters
TOTAL_CAP_GB=1000
NUM_LANGS=200

# 1. Inventory (only if needed, usually fast)
if [ ! -f "$INVENTORY_FILE" ]; then
    echo "Generating inventory..."
    python ${SAMPLING_SCRIPTS}/inventory_datasets.py
fi

# 2. Calculate Quotas
echo "Calculating quotas for ${TOTAL_CAP_GB}GB and top ${NUM_LANGS} languages..."
python ${SAMPLING_SCRIPTS}/calculate_quotas.py \
    --cap_gb ${TOTAL_CAP_GB} \
    --num_langs ${NUM_LANGS} \
    --inventory ${INVENTORY_FILE} \
    --output ${QUOTAS_FILE}

# 3. Download and Sample
echo "Starting download and sampling..."
mkdir -p ${OUTPUT_ROOT}

python ${SAMPLING_SCRIPTS}/sample_and_download.py \
    --quotas ${QUOTAS_FILE} \
    --inventory ${INVENTORY_FILE} \
    --output ${SAMPLED_DATA_DIR} \
    --tokenizer_data ${TOKENIZER_DATA} \
    --workers 16

echo "Job finished at $(date)"
