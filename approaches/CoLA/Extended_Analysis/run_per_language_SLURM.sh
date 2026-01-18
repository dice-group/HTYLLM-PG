#!/bin/bash
#SBATCH --job-name=routing_per_lang
#SBATCH --output=logs/routing_lang_%A_%a.out
#SBATCH --error=logs/routing_lang_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-9  # Adjust based on number of languages

# Parallel Expert Routing Analysis - Per-Language Job Array
# This script processes one language per job for maximum parallelization

set -e

echo "========================================="
echo "Per-Language Routing Analysis (Slurm)"
echo "========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate environment
# source activate your_env_name

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B"
CHECKPOINT="/scratch/hpc-prf-merlin/sashreek/moe_study/saves/hydralora_moe_llama31_8b_acc"
DATA_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples"
OUTPUT_BASE="./analysis"
# Note: --adapter_type, --num_layers, --num_experts are auto-detected from adapter_config.json

# Get list of languages from data directory
LANGUAGES=($(ls ${DATA_DIR}/*.jsonl | xargs -n1 basename | sed 's/.jsonl//'))
NUM_LANGS=${#LANGUAGES[@]}

echo "Total languages: $NUM_LANGS"
echo "Processing task: $SLURM_ARRAY_TASK_ID"

# Check if this task is within bounds
if [ $SLURM_ARRAY_TASK_ID -ge $NUM_LANGS ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of languages ($NUM_LANGS), exiting"
    exit 0
fi

# Get language for this task
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}
echo "Analyzing language: $LANG"
echo ""

# Create output directory
mkdir -p "${OUTPUT_BASE}/per_language"

# Run analysis for this language only (adapter_type, num_layers, num_experts auto-detected)
srun python tool/analyze_expert_routing.py \
    --base_model "$BASE_MODEL" \
    --adapter_checkpoint "$CHECKPOINT" \
    --test_data "$DATA_DIR" \
    --languages "$LANG" \
    --output "${OUTPUT_BASE}/per_language/${LANG}" \
    --batch_size 8 \
    --device cuda

echo ""
echo "✓ Completed analysis for language: $LANG"
echo "Finished at: $(date)"
