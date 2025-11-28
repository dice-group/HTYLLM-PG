#!/bin/bash
#SBATCH --job-name=convert-moe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=convert_%j.log

set -e

# ---------- Env ----------
source ~/.bashrc
conda activate moe

module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p /scratch/hpc-prf-merlin/luke/.cache/torch_extensions
mkdir -p /scratch/hpc-prf-merlin/luke/.triton/autotune
export TORCH_EXTENSIONS_DIR="/scratch/hpc-prf-merlin/luke/.cache/torch_extensions"
export TRITON_CACHE_DIR="/scratch/hpc-prf-merlin/luke/.triton/autotune"
export XDG_CACHE_HOME="/scratch/hpc-prf-merlin/luke/.cache"

# Base directories
CHECKPOINT_ROOT="../checkpoints"
OUTPUT_ROOT="../hf_models"
CONFIG_PATH="config_3_7b.json"

# Ensure output directory exists
mkdir -p "$OUTPUT_ROOT"

# Find all step_* directories in the checkpoint root
# Adjust the pattern if your structure is different
for ckpt_dir in "$CHECKPOINT_ROOT"/step_*; do
    if [ -d "$ckpt_dir" ]; then
        step_name=$(basename "$ckpt_dir")
        output_dir="$OUTPUT_ROOT/$step_name"
        
        echo "=================================================="
        echo "Converting $step_name..."
        echo "  Input:  $ckpt_dir"
        echo "  Output: $output_dir"
        echo "=================================================="
        
        # Check if already converted
        if [ -f "$output_dir/config.json" ]; then
             echo "Output config.json exists. Skipping $step_name (remove folder to re-convert)."
             continue
        fi

        deepspeed convert_ds_to_hf.py \
            --checkpoint_path "$ckpt_dir" \
            --output_dir "$output_dir" \
            --config_path "$CONFIG_PATH"
            
        echo "Finished converting $step_name."
        echo ""
    fi
done

echo "All conversions complete!"

