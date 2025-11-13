#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=./LLaMA-Factory/saves/smoke_test

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --resize_vocab true \
  --dataset gsm8k \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type hydralora \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --bf16 True \
  --fp16 False \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_num 1 \
  --use_hydralora_experts \
  --hydralora_num_experts 2 \
  --hydralora_top_k 2 \
  --hydralora_debug  \
  2>&1 | tee train_moe_hydralora_debug.log

: '
This script is the HydraLoRA MoE counterpart to train_moe_cola.sh.

Key parallels:
- Same base model + dataset as CoLA MoE (llama-3.2-1B-multilingual + c4).
- MoE toggles:
    CoLA:  --use_cola_experts      -> Hydra: --use_hydralora_experts
           --cola_num_experts      -> Hydra: --hydralora_num_experts
           --cola_top_k            -> Hydra: --hydralora_top_k
           --cola_debug            -> Hydra: --hydralora_debug
- Same training regime (epochs, batch size, bf16).

Core HydraLoRA remains:
- One A, multiple B per (expert) adapter (controlled by --lora_num).
- No PiSSA init; just standard HydraLoRA initialization.
'
