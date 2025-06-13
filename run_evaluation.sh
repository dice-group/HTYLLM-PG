#!/bin/bash

#SBATCH --job-name=gpt2-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin

# LM Evaluation Harness runner script for custom GPT-2 model
# Usage: sbatch run_evaluation.sh [model_path] [tasks] [additional_args]

set -e

# Activate environment
source ~/miniconda3/bin/activate icebreaker

module load system/CUDA/12.6.0

# Default values
MODEL_PATH="${1:-gpt2_model_step_49000.pt}"
TASKS="${2:-belebele}"
TOKENIZER_PATH="tokenizer/sp_model_131072/sp_model_131072.model"
OUTPUT_DIR="evaluation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "LM Evaluation Harness - Custom GPT-2"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo "Available model files:"
    ls -la *.pt 2>/dev/null || echo "No .pt files found in current directory"
    exit 1
fi

# Check if tokenizer exists
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer file not found: $TOKENIZER_PATH"
    exit 1
fi

# Check if lm_eval is installed
if ! python -c "import lm_eval" 2>/dev/null; then
    echo "ERROR: lm_eval not installed. Installing..."
    pip install -r requirements_eval.txt
fi

# Run evaluation
echo "Starting evaluation..."
python src/model/lm_eval_harness.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --tasks "$TASKS" \
    --output_path "$OUTPUT_DIR/results_${TIMESTAMP}.json" \
    --device auto \
    --batch_size 1 \
    --num_fewshot 0 \
    "${@:3}"  # Pass any additional arguments

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/results_${TIMESTAMP}.json"
echo "=========================================="

# Quick summary
if [ -f "$OUTPUT_DIR/results_${TIMESTAMP}.json" ]; then
    echo "Quick summary:"
    python -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/results_${TIMESTAMP}.json', 'r') as f:
        results = json.load(f)
    
    print('Task Results:')
    for task, metrics in results.get('results', {}).items():
        print(f'  {task}:')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'    {metric}: {value:.4f}')
except Exception as e:
    print(f'Error reading results: {e}')
"
fi 