#!/bin/bash
#SBATCH --job-name=lm-harness
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=lm_harness_eval_%j.out

echo "Loading modules..."
module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0

# Activate the Conda environment
echo "Activating Conda environment..."
source ~/miniconda3/bin/activate meg

# --- Configuration ---
# Set the Hugging Face model identifier
MODEL_NAME="LckyLke/moe_tf_model"

OUTPUT_DIR="./results/${MODEL_NAME}"

TASK_LIST="hellaswag,xnli,belebele,arc_multilingual,global_mmlu,include_base_44,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa,xcopa,xwinograd,xstorycloze,pawsx,flores,wmt16,lambada_multilingual,xquad"

# --- Sanity Checks and Information ---
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "---"
echo "Model: $MODEL_NAME"
echo "Tasks: $TASK_LIST"
echo "Output Directory: $OUTPUT_DIR"
echo "---"

# Display GPU status
nvidia-smi

# --- Run Evaluation ---
echo "Starting lm-evaluation-harness..."

# The main command to run the evaluation
lm_eval --model hf \
    --model_args pretrained=${MODEL_NAME},trust_remote_code=True,dtype=bfloat16 \
    --tasks ${TASK_LIST} \
    --device cuda:0 \
    --batch_size auto \
    --output_path ${OUTPUT_DIR} \
    --log_samples

# --- Job Completion ---
echo "Evaluation finished."
echo "Results saved in ${OUTPUT_DIR}"
echo "Job finished at: $(date)"