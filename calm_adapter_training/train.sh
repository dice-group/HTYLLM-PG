#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate htyllm

accelerate launch --config_file accelerate_config.yaml train.py \
  --anchor_model_dir google/gemma-7b \
  --aug_model_dir google/gemma-2b \
  --num_heads 2 \
  --num_connections 2 \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --output_dir '/data/joel/calm-results/gemma2_7b'
