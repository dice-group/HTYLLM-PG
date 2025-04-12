#!/bin/bash

export HF_HOME=/data/hf_cache/
export INPUT_DIR="/data/joel/prepared/realnewslike/"

MODEL_NAME="gpt2"
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-5
OUTPUT_MODEL_DIR="./model_output"

python3 train_test.py \
    --processed_dir "$INPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --output_model_dir "$OUTPUT_MODEL_DIR"

echo "Training finished."
