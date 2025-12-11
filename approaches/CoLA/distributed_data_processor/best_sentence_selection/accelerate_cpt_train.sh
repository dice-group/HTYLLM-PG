#!/usr/bin/env bash
#SBATCH --job-name=cpt-accelerate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/cpt_accel_%j.log
#SBATCH --partition=gpu

set -euo pipefail
module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

# ---- conda / environment -------------------------------------------------
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cola_llama_factory

# ---- sanity check --------------------------------------------------------
python -c "import torch, transformers; \
          print('CUDA available:', torch.cuda.is_available()); \
          print('GPU count   :', torch.cuda.device_count()); \
          print('Transformers version:', transformers.__version__)"


# -----------------------------------------------------------------------
#  USER‑CONFIGURABLE PARAMETERS (edit them once)
# -----------------------------------------------------------------------
BASE_MODEL="meta-llama/Llama-3.1-8B"           # same model you used before
CPT_CORPUS="/scratch/hpc-prf-merlin/yven/cpt/Llama-3.1-8B/eng_plus_5_langs/cpt_corpus_10000.jsonl"  # output from cpt_preparation_pipeline.sh
OUT_DIR="/scratch/hpc-prf-merlin/yven/cpt/Llama-3.1-8B/eng_plus_5_langs/checkpoint_cpt_10k"
#TOKENIZER_DIR="./augmented_tokenizer"       # <-- set to path of the tokenizer you built;
                                            #     if you didn’t augment, comment the line
EPOCHS=3
BATCH_SIZE=16
LR=3e-5
SEED=42

# -----------------------------------------------------------------------
#  ACCELERATE SETTINGS
# -----------------------------------------------------------------------
ACCEL_CONFIG=../LLaMA-Factory/examples/accelerate/single_gpu.yaml

# -----------------------------------------------------------------------
#  RUN THE TRAINING
# -----------------------------------------------------------------------
echo "[INFO] Starting accelerate-CPT training at $(date)"

accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  ./cpt_train.py \
    --base_model "$BASE_MODEL" \
    --cpt_corpus "$CPT_CORPUS" \
    --output_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --seed "$SEED" \
#$( [ -d "$TOKENIZER_DIR" ] && echo "--tokenizer_dir $TOKENIZER_DIR" ) \

echo "[INFO] Finished - checkpoint saved in $OUT_DIR"