#!/bin/bash

# Environment setup
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1

# Configurable parameters
TOKENIZED_DIR="/data/joel/tokenized_adapter_subsets/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian"
TOKENIZER="mistralai/Mistral-7B-v0.3"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
OUTPUT_DIR="/data/joel/results_language_adapters/xlora/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian"
LOGGING_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGGING_DIR"

TRAIN_BATCH_SIZE=10
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
RUN_NAME="mistral-7b_xlora_asian_swh_sna"
DATALOADER_NUM_WORKERS=1
EVALUATION_STRATEGY="steps"
EVAL_STEPS=501
LOAD_BEST_MODEL_AT_END=true
METRIC_FOR_BEST_MODEL="eval_accuracy"
GREATER_IS_BETTER=true

EVAL_INTERVAL=501
#EVAL_TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
EVAL_TASKS="belebele_swh_Latn,belebele_sna_Latn,belebele_nya_Latn,include_base_44_vietnamese,belebele_vie_Latn,belebele_tha_Thai,belebele_jav_Latn,belebele_sun_Latn,belebele_khm_Khmr"
EVAL_METRICS_EARLYSTOPPING="belebele_swh_Latn,belebele_sna_Latn,belebele_nya_Latn,include_base_44_vietnamese,belebele_vie_Latn,belebele_tha_Thai,belebele_jav_Latn,belebele_sun_Latn,belebele_khm_Khmr"
EARLY_STOPPING_PATIENCE=3
RESUME_FROM_CHECKPOINT=True

EVAL_BATCH_SIZE=10
EVAL_LIMIT=200
EVAL_CUDA_DEVICES="2"
export CUDA_VISIBLE_DEVICES="$EVAL_CUDA_DEVICES"
EVAL_LOG_SAMPLES=true
EVAL_WANDB_PROJECT="mistral-7b_xlora_asian_swh_sna"
LOG_FILE="$OUTPUT_DIR/training.log"

# Run training
nohup python train_xlora_adapter_enhanced.py \
  --tokenized_dir "$TOKENIZED_DIR" \
  --tokenizer_path "$TOKENIZER" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
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