#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=./LLaMA-Factory/saves/smoke_test

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --dataset gsm8k \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type cola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --num_A 1 \
  --num_B 1 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --use_cola_experts \
  --cola_num_experts 4 \
  --cola_top_k 2 \
  --cola_debug 2>&1 | tee train_moe_debug.log

: '
TODOs:
- use multilingual data: sample data as in cluster or run in cluster
- use multilingual tokenizer (maybe do one run with and without to compare complexity)
- preprocess data with llamafactory
- run longer MoE cola test
'