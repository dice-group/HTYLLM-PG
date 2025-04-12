#!/bin/bash

export OUTPUT_DIR="/data/joel/output_realnews"
mkdir -p $OUTPUT_DIR

MODEL_NAME="facebook/opt-350m"
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-5
OUTPUT_MODEL_DIR="./model_output"

python3 train_test.py \
    --processed_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --output_model_dir "$OUTPUT_MODEL_DIR"

echo "Training finished."
