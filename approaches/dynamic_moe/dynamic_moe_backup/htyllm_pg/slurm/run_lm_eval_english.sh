#!/bin/bash
#SBATCH --job-name=eval-english
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=eval_english_%j.log

set -e

# ---------- Env ----------
source ~/.bashrc
conda activate moe

module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0

# Directory containing the converted HF models (subdirectories for each step)
HF_MODELS_DIR="hf_models"
RESULTS_DIR="eval_results"

# English Tasks
TASKS="hellaswag,winogrande,piqa,arc_easy,arc_challenge,boolq,openbookqa"

BATCH_SIZE="auto"

# Ensure output directory exists
mkdir -p "$RESULTS_DIR"

echo "=================================================="
echo "Starting Batch Evaluation on English Tasks"
echo "Models Dir: $HF_MODELS_DIR"
echo "Tasks:      $TASKS"
echo "=================================================="

# Loop through each model folder
for model_dir in "$HF_MODELS_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        output_file="$RESULTS_DIR/${model_name}_english_results.json"

        echo "--------------------------------------------------"
        echo "Evaluating: $model_name"
        echo "Path:       $model_dir"
        
        # Skip if results already exist
        if [ -f "$output_file" ]; then
            echo "Results already exist at $output_file. Skipping..."
            continue
        fi

        # Ensure tokenizer is present in model dir (using absolute path for tokenizer source)
        cp tokenizer.json "$model_dir/" || echo "Warning: Could not copy tokenizer.json"
        cp tokenizer_config.json "$model_dir/" || echo "Warning: Could not copy tokenizer_config.json"


        # Run lm_eval
        lm_eval --model hf \
            --model_args pretrained="$model_dir",trust_remote_code=True,dtype=bfloat16 \
            --tasks $TASKS \
            --batch_size $BATCH_SIZE \
            --device cuda:0 \
            --output_path "$output_file" \
            --log_samples

        echo "Finished $model_name"
    fi
done

echo "=================================================="
echo "All evaluations complete!"

