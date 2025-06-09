#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")/.."

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

# Run multi-GPU training with torchrun
# --nproc_per_node=2 to use both GPUs
torchrun \
  --nproc_per_node=2 \
  src/train.py \
  --model_path checkpoints/init \
  --tokenizer_path tokenizer \
  --dataset_dir data/processed \
  --output_dir checkpoints/train_run2_vm \
  --deepspeed_config src/configs/deepspeed/ds_zero3_moe.json \
  --batch_size 32 \
  --grad_accum 4

echo "Training complete!"
