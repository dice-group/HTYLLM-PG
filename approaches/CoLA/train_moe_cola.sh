#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR=./LLaMA-Factory/data
OUTPUT_DIR=./LLaMA-Factory/saves/smoke_test

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path data/project_data/moe_study/models/llama-3.2-1B-multilingual \
  --resize_vocab true \
  --dataset c4 \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --finetuning_type cola \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --num_A 1 \
  --num_B 1 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --use_cola_experts \
  --cola_num_experts 2 \
  --cola_top_k 2 \
  --bf16 True \
  --fp16 False \
  --cola_debug 2>&1 | tee train_moe_debug.log

: '
TODOs:
- use multilingual data: sample data as in cluster or run in cluster
- use multilingual tokenizer (maybe do one run with and without to compare complexity)
- preprocess data with llamafactory
- run longer MoE cola test
'
 scp joeldag@login4.ln2025.pc2.uni-paderborn.de:/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/merged/state.json  /data/project_data/moe_study/tokenized/test_llama_3.2-1B_multilingual_tok/