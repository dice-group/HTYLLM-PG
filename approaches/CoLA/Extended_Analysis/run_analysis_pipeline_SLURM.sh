#!/bin/bash
#SBATCH --job-name=expert_routing_analysis
#SBATCH --output=logs/routing_analysis_%j.out
#SBATCH --error=logs/routing_analysis_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

# Expert Routing Analysis - Slurm Version
# This script is optimized for HPC clusters with Slurm workload manager

set -e  # Exit on error

echo "========================================="
echo "Expert Routing Analysis (Slurm)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""


set -euo pipefail

module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

# Use shared scratch HF cache
export HF_HOME=/scratch/hpc-prf-merlin/shared_cache/huggingface/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate hydralora_llama_factory
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
# Disabled: torchrun handles GPU assignment automatically
# if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
#   export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
# fi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"


# Configuration from command line or defaults
BASE_MODEL="meta-llama/Llama-3.1-8B"
CHECKPOINT="/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_llama31_8b_acc/pissa/checkpoint-6000"
ADAPTER_TYPE="hydralora"
VALIDATION_DATA="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples"
LANGUAGES="en,es,hi,ru,fi"
NUM_SEQUENCES="10000"
NUM_LAYERS="32"
NUM_EXPERTS="4"
BATCH_SIZE="8"

# Derived paths
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
OUTPUT_DIR="./analysis/${CHECKPOINT_NAME}"
DATA_DIR="./data/language_test_sets"
LOGS_DIR="./logs"

# Create logs directory
mkdir -p "$LOGS_DIR"

echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Adapter Type: $ADAPTER_TYPE"
echo "  Languages: $LANGUAGES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Prepare test data (skip if already exists)
if [ ! -d "$DATA_DIR" ]; then
    echo "[1/5] Preparing language test datasets..."
    srun python tool/prepare_language_datasets.py \
        --validation_data "$VALIDATION_DATA" \
        --languages "$LANGUAGES" \
        --num_sequences "$NUM_SEQUENCES" \
        --output_dir "$DATA_DIR" \
        2>&1 | tee "$LOGS_DIR/step1_prepare_data_${SLURM_JOB_ID}.log"
    echo ""
else
    echo "[1/5] Skipping data preparation (already exists)"
    echo ""
fi

# Step 2: Analyze routing
echo "[2/5] Running expert routing analysis..."
srun python tool/analyze_expert_routing.py \
    --base_model "$BASE_MODEL" \
    --adapter_checkpoint "$CHECKPOINT" \
    --adapter_type "$ADAPTER_TYPE" \
    --test_data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --num_layers "$NUM_LAYERS" \
    --num_experts "$NUM_EXPERTS" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    2>&1 | tee "$LOGS_DIR/step2_analyze_${SLURM_JOB_ID}.log"
echo ""

# Step 3: Normalize data
echo "[3/5] Applying layer-wise normalization..."
srun python tool/process_routing_data.py \
    --input "$OUTPUT_DIR/routing_matrix.npz" \
    --output "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    2>&1 | tee "$LOGS_DIR/step3_normalize_${SLURM_JOB_ID}.log"
echo ""

# Step 4: Generate visualizations
echo "[4/5] Creating visualizations..."
srun python tool/visualize_expert_routing.py \
    --routing_data "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    --language_families ./config/language_families.json \
    --output_dir "$OUTPUT_DIR/figures" \
    --create_all \
    2>&1 | tee "$LOGS_DIR/step4_visualize_${SLURM_JOB_ID}.log"
echo ""

# Step 5: Generate report
echo "[5/5] Generating analysis report..."
srun python tool/generate_analysis_report.py \
    --routing_data "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    --language_families ./config/language_families.json \
    --figures_dir "$OUTPUT_DIR/figures" \
    --output "$OUTPUT_DIR/report.md" \
    2>&1 | tee "$LOGS_DIR/step5_report_${SLURM_JOB_ID}.log"
echo ""

echo "========================================="
echo "Analysis Complete!"
echo "========================================="
echo "Finished at: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View report: $OUTPUT_DIR/report.md"
echo "View figures:"
echo "  - Heatmap: $OUTPUT_DIR/figures/routing_heatmap.png"
echo "  - t-SNE: $OUTPUT_DIR/figures/tsne_clustering.png"
echo "  - Entropy: $OUTPUT_DIR/figures/layer_entropy.png"
echo ""
echo "Job completed successfully!"
