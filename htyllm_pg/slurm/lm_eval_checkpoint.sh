#!/bin/bash
#SBATCH --job-name=moe-lm-eval
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/lm_eval_%x_%j.log             #verify log path

set -e

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)   CKPT=$2; shift 2;;
    --tokenizer)    TOK=$2; shift 2;;
    --tasks)        TASKS=$2; shift 2;;
    --batch-size)   BS=$2; shift 2;;
    --output-dir)   OUTDIR=$2; shift 2;;
    --wandb-project) WP=$2; shift 2;;
    --wandb-prefix)  PREF=$2; shift 2;;
    --extra-args)    EXTRA=$2; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

# params
TASKS=${TASKS:-belebele}
BS=${BS:-auto}
TOK=${TOK:-$CKPT}

[[ -z "$CKPT" || -z "$OUTDIR" ]] && { echo "--checkpoint and --output-dir required"; exit 1; }
[[ ! -d "$CKPT" ]] && { echo "Checkpoint not found: $CKPT"; exit 1; }

mkdir -p "$OUTDIR" logs
module purge
module load toolchain/foss/2024a system/CUDA/12.6.0 lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0    #TODO verify that these work
source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate test2

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
export PYTHONUNBUFFERED=1

# ---- Run lm_eval ----
LABEL=$(basename "$CKPT")
OUTFILE="${OUTDIR}/${LABEL}_lm_eval.jsonl"

echo "Running lm_eval on $LABEL..."

lm_eval \
  --model hf \
  --model_args "pretrained=$CKPT,tokenizer=$TOK" \
  --tasks "$TASKS" \
  --batch_size "$BS" \
  --output_path "$OUTFILE" \
  $EXTRA

echo "Done!"