#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate htyllm

accelerate launch --config_file accelerate_config.yaml train.py \
  --anchor_model_dir google/gemma-7b \
  --aug_model_dir google/gemma-3-4b-it \
  --num_heads 2 \
  --num_connections 2 \
  --learning_rate 3e-4 \
  --batch_size 2 \
  --output_dir '/data/joel/calm-results/gemma2_7b'
