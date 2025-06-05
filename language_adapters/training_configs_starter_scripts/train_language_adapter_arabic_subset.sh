#!/bin/bash

# Environment setup
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0

# Configurable parameters
TOKENIZED_DIR="/data/joel/tokenized_adapter_subsets/gemma3-4b-pt/arabic_subset/"
TOKENIZER_PATH="/data/joel/tokenized_adapter_subsets/gemma_extended_tokenizers/arabic/"
MODEL_NAME="google/gemma-3-4b-pt"
OUTPUT_DIR="/data/joel/results_language_adapters/gemma3-4b-pt/arabic"
LOGGING_DIR="/data/joel/results_language_adapters/gemma3-4b-pt/arabic/logs"

LOAD_IN_4BIT=true
BNB_4BIT_USE_DOUBLE_QUANT=true
BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_COMPUTE_DTYPE="float16"

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_BIAS="none"
LORA_TARGET_MODULES="q_proj,v_proj,gate_proj,up_proj"

TRAIN_BATCH_SIZE=52
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=2
LEARNING_RATE=2e-4
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=20
SAVE_STRATEGY="steps"
SAVE_STEPS=1000
BF16=true
SAVE_TOTAL_LIMIT=200
REPORT_TO="wandb,tensorboard"
RUN_NAME="gemma3-4b-pt-arabic-adapter"
DATALOADER_NUM_WORKERS=4
EVALUATION_STRATEGY="steps"
EVAL_STEPS=1000
LOAD_BEST_MODEL_AT_END=true
METRIC_FOR_BEST_MODEL="eval_accuracy"
GREATER_IS_BETTER=true

EVAL_INTERVAL=1001
EVAL_TASKS="belebele"
EVAL_METRICS_EARLYSTOPPING="belebele_apc_Arab,belebele_ary_Arab,belebele_arz_Arab,belebele_ars_Arab,belebele_heb_Hebr"
EARLY_STOPPING_PATIENCE=3
RESUME_FROM_CHECKPOINT=false

EVAL_BATCH_SIZE=4
EVAL_LIMIT=500
EVAL_CUDA_DEVICES="0,1"
EVAL_LOG_SAMPLES=true
EVAL_WANDB_PROJECT="gemma3-4b-pt-arabic-lanauge-adapter"


# Run training
python ../train_language_adapter.py \
  --tokenized_dir "$TOKENIZED_DIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
  --load_in_4bit "$LOAD_IN_4BIT" \
  --bnb_4bit_use_double_quant "$BNB_4BIT_USE_DOUBLE_QUANT" \
  --bnb_4bit_quant_type "$BNB_4BIT_QUANT_TYPE" \
  --bnb_4bit_compute_dtype "$BNB_4BIT_COMPUTE_DTYPE" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_bias "$LORA_BIAS" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy "$SAVE_STRATEGY" \
  --save_steps "$SAVE_STEPS" \
  --bf16 "$BF16" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --report_to "$REPORT_TO" \
  --run_name "$RUN_NAME" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --evaluation_strategy "$EVALUATION_STRATEGY" \
  --eval_steps "$EVAL_STEPS" \
  --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END" \
  --metric_for_best_model "$METRIC_FOR_BEST_MODEL" \
  --greater_is_better "$GREATER_IS_BETTER" \
  --eval_interval "$EVAL_INTERVAL" \
  --eval_tasks "$EVAL_TASKS" \
  --eval_metric_names "$EVAL_METRICS_EARLYSTOPPING" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --eval_limit $EVAL_LIMIT \
  --eval_cuda_devices "$EVAL_CUDA_DEVICES" \
  --eval_wandb_project "$EVAL_WANDB_PROJECT"
