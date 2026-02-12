#!/bin/bash
#SBATCH --job-name=group_routing_analysis
#SBATCH --output=logs/group_analysis_%j.out
#SBATCH --error=logs/group_analysis_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

# Group-level Expert Routing Analysis - Slurm Version
# This script analyzes routing patterns aggregated by language families or subgroups
# It operates on pre-computed routing data and does NOT require GPU

set -e  # Exit on error

echo "========================================="
echo "Group-level Routing Analysis (Slurm)"
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
# INPUT: Path to the routing_matrix_normalized.npz from the main analysis
ROUTING_DATA="/scratch/hpc-prf-merlin/project_data/moe_study/extended_analysis/samples_100/cola_colaexp-lpr_20260108_055323/checkpoint-110000_adapter/routing_matrix_normalized.npz"

# GROUPINGS: Path to the language groupings JSON file
GROUPINGS_FILE="/pc2/users/s/sashreek/HTYLLM-PG/approaches/CoLA/tools/two_stage_clustering/200_tier_language_groupings.json"

# AGGREGATION_LEVEL: What level to aggregate by ("families" or "subgroups")
# AGGREGATION_LEVEL="families"
AGGREGATION_LEVEL="subgroups"

# Derived paths
INPUT_DIR=$(dirname "$ROUTING_DATA")
INPUT_BASENAME=$(basename "$ROUTING_DATA" .npz)
OUTPUT_DIR="${INPUT_DIR}/group_analysis_${AGGREGATION_LEVEL}"
LOGS_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/extended_analysis/logs"

# Create logs directory
mkdir -p "$LOGS_DIR"

echo "Configuration:"
echo "  Input Routing Data: $ROUTING_DATA"
echo "  Groupings File: $GROUPINGS_FILE"
echo "  Aggregation Level: $AGGREGATION_LEVEL"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Validation: Check if input file exists
if [ ! -f "$ROUTING_DATA" ]; then
    echo "ERROR: Routing data file not found: $ROUTING_DATA"
    echo "Please run the main analysis pipeline first (run_analysis_pipeline_SLURM.sh)"
    exit 1
fi

if [ ! -f "$GROUPINGS_FILE" ]; then
    echo "ERROR: Groupings file not found: $GROUPINGS_FILE"
    exit 1
fi

# Run group-level analysis
echo "[1/1] Running group-level routing analysis..."
srun python tool/analyze_by_groups.py \
    --input "$ROUTING_DATA" \
    --groupings "$GROUPINGS_FILE" \
    --output "$OUTPUT_DIR" \
    --aggregate_by "$AGGREGATION_LEVEL" \
    2>&1 | tee "$LOGS_DIR/group_analysis_${SLURM_JOB_ID}.log"
echo ""

echo "========================================="
echo "Group Analysis Complete!"
echo "========================================="
echo "Finished at: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View report: $OUTPUT_DIR/report.md"
echo "View figures:"
echo "  - Heatmap: $OUTPUT_DIR/figures/routing_heatmap.png"
echo ""
echo "Job completed successfully!"
