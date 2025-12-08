#!/bin/bash
# Complete pipeline script for expert routing analysis
# This script runs the entire analysis workflow from start to finish

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Expert Routing Analysis Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B"
CHECKPOINT="/scratch/hpc-prf-merlin/sashreek/moe_study/saves/hydralora_moe_llama31_8b_acc"
ADAPTER_TYPE="hydralora"
VALIDATION_DATA="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples"
LANGUAGES="en,es,fr,de,zh"
NUM_SEQUENCES="10000"
NUM_LAYERS="32"
NUM_EXPERTS="4"

# Derived paths
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
OUTPUT_DIR="./analysis/${CHECKPOINT_NAME}"
DATA_DIR="./data/language_test_sets"

echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Adapter Type: $ADAPTER_TYPE"
echo "  Languages: $LANGUAGES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Prepare test data (skip if already exists)
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${GREEN}[1/5] Preparing language test datasets...${NC}"
    python tool/prepare_language_datasets.py \
        --validation_data "$VALIDATION_DATA" \
        --languages "$LANGUAGES" \
        --num_sequences "$NUM_SEQUENCES" \
        --output_dir "$DATA_DIR"
    echo ""
else
    echo -e "${GREEN}[1/5] Skipping data preparation (already exists)${NC}"
    echo ""
fi

# Step 2: Analyze routing
echo -e "${GREEN}[2/5] Running expert routing analysis...${NC}"
python tool/analyze_expert_routing.py \
    --base_model "$BASE_MODEL" \
    --adapter_checkpoint "$CHECKPOINT" \
    --adapter_type "$ADAPTER_TYPE" \
    --test_data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --num_layers "$NUM_LAYERS" \
    --num_experts "$NUM_EXPERTS" \
    --batch_size 16
echo ""

# Step 3: Normalize data
echo -e "${GREEN}[3/5] Applying layer-wise normalization...${NC}"
python tool/process_routing_data.py \
    --input "$OUTPUT_DIR/routing_matrix.npz" \
    --output "$OUTPUT_DIR/routing_matrix_normalized.npz"
echo ""

# Step 4: Generate visualizations
echo -e "${GREEN}[4/5] Creating visualizations...${NC}"
python tool/visualize_expert_routing.py \
    --routing_data "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    --language_families ./config/language_families.json \
    --output_dir "$OUTPUT_DIR/figures" \
    --create_all
echo ""

# Step 5: Generate report
echo -e "${GREEN}[5/5] Generating analysis report...${NC}"
python tool/generate_analysis_report.py \
    --routing_data "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    --language_families ./config/language_families.json \
    --figures_dir "$OUTPUT_DIR/figures" \
    --output "$OUTPUT_DIR/report.md"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Analysis Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View report: $OUTPUT_DIR/report.md"
echo "View figures:"
echo "  - Heatmap: $OUTPUT_DIR/figures/routing_heatmap.png"
echo "  - t-SNE: $OUTPUT_DIR/figures/tsne_clustering.png"
echo "  - Entropy: $OUTPUT_DIR/figures/layer_entropy.png"
