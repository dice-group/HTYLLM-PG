#!/bin/bash


# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

# Run multi-GPU training with torchrun
# --nproc_per_node=2 to use both GPUs
torchrun \
  --nproc_per_node=2 \
  src/train.py \
  --model_path checkpoints/pretrain_run_vm/checkpoint-16000 \
  --tokenizer_path tokenizer \
  --dataset_dir data/processed \
  --output_dir checkpoints/pretrain_run_vm \
  --deepspeed_config src/configs/deepspeed/ds_zero3_moe.json \
  --batch_size 12 \
  --grad_accum 4

echo "Training complete!"
