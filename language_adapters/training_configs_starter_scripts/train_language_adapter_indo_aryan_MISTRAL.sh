#!/bin/bash

# Environment setup
export WANDB_MODE=online

# Configurable parameters
TOKENIZED_DIR="/upb/users/j/joeldag/profiles/unix/cs/tokenized_data_subsets/mistral7b/indo_aryan"
TOKENIZER_PATH="/upb/users/j/joeldag/profiles/unix/cs/tokenized_data_subsets/mistral_extended_tokenizers"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
OUTPUT_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/mistral7b/indo_aryan"
LOGGING_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/mistral7b/indo_aryan/logs"
mkdir -p "$LOGGING_DIR"

LOAD_IN_4BIT=true
BNB_4BIT_USE_DOUBLE_QUANT=true
BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_COMPUTE_DTYPE="float16"

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_BIAS="none"
LORA_TARGET_MODULES="q_proj,v_proj,gate_proj,up_proj"

TRAIN_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-4
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=20
SAVE_STRATEGY="steps"
SAVE_STEPS=500
BF16=true
SAVE_TOTAL_LIMIT=200
REPORT_TO="wandb,tensorboard"
RUN_NAME="mistral7b-indo_aryan-adapter"
DATALOADER_NUM_WORKERS=1
EVALUATION_STRATEGY="steps"
EVAL_STEPS=501 #glaube das ist überflüssig wegen meinem lmharnes eval
LOAD_BEST_MODEL_AT_END=true
METRIC_FOR_BEST_MODEL="eval_accuracy"
GREATER_IS_BETTER=true

EVAL_INTERVAL=501
EVAL_TASKS="include_base_44_hindi,belebele_hin_Deva,belebele_urd_Arab,include_base_44_urdu,belebele_mar_Deva,include_base_44_bengali,belebele_ben_Beng,belebele_pan_Guru"
EVAL_METRICS_EARLYSTOPPING="include_base_44_hindi,belebele_hin_Deva,belebele_urd_Arab,include_base_44_urdu,belebele_mar_Deva,include_base_44_bengali,belebele_ben_Beng,belebele_pan_Guru"
EARLY_STOPPING_PATIENCE=3
RESUME_FROM_CHECKPOINT=False

EVAL_BATCH_SIZE=10
EVAL_LIMIT=200
EVAL_CUDA_DEVICES="0"
export CUDA_VISIBLE_DEVICES="$EVAL_CUDA_DEVICES"
EVAL_LOG_SAMPLES=true
EVAL_WANDB_PROJECT="mistral7b-indo-aryan-language-adapter"
LOG_FILE="$OUTPUT_DIR/training.log"

# Run training
nohup python ../train_language_adapter.py \
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
  --eval_wandb_project "$EVAL_WANDB_PROJECT" > "$LOG_FILE" 2>&1 &

sleep 2
echo "Training started with PID $!"
echo "Logging to: $LOG_FILE"
tail -f "$LOG_FILE"
 