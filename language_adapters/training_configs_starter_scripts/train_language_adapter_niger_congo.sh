#!/bin/bash

# Environment setup
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1

# Configurable parameters
TOKENIZED_DIR="/upb/users/j/joeldag/profiles/unix/cs/tokenized_data_subsets/gemma3-4b-pt/niger_congo"
TOKENIZER_PATH="/upb/users/j/joeldag/profiles/unix/cs/tokenized_data_subsets/gemma_extended_tokenizers/niger_congo/"
MODEL_NAME="google/gemma-3-4b-pt"
OUTPUT_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/gemma3-4b-pt/niger_congo"
LOGGING_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/gemma3-4b-pt/niger_congo/logs"

LOAD_IN_4BIT=true
BNB_4BIT_USE_DOUBLE_QUANT=true
BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_COMPUTE_DTYPE="float16"

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_BIAS="none"
LORA_TARGET_MODULES="q_proj,v_proj,gate_proj,up_proj"

TRAIN_BATCH_SIZE=20
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
RUN_NAME="gemma3-4b-pt-niger-congo-adapter"
DATALOADER_NUM_WORKERS=1
EVALUATION_STRATEGY="steps"
EVAL_STEPS=1001 #glaube das ist überflüssig wegen meinem lmharnes eval
LOAD_BEST_MODEL_AT_END=true
METRIC_FOR_BEST_MODEL="eval_accuracy"
GREATER_IS_BETTER=true

EVAL_INTERVAL=1001
EVAL_TASKS="belebele_swh_Latn,belebele_yor_Latn,belebele_ibo_Latn,belebele_wol_Latn,belebele_zul_Latn,afrimgsm_direct_yor,afrimgsm_en_cot_yor,afrimgsm_translate_direct_yor,afrimmlu_direct_yor,afrixnli_en_direct_yor,afrixnli_manual_direct_yor,afrixnli_manual_translate_yor,afrixnli_native_direct_yor,afrixnli_translate_yor,afrimgsm_direct_ibo,afrimgsm_en_cot_ibo,afrimgsm_translate_direct_ibo,afrimmlu_direct_ibo,afrimmlu_translate_ibo,afrixnli_en_direct_ibo,afrixnli_manual_direct_ibo,afrixnli_manual_translate_ibo,afrixnli_translate_ibo,afrixnli_native_direct_ibo,afrimgsm_direct_wol,afrimgsm_en_cot_wol,afrimgsm_translate_direct_wol,afrimmlu_direct_wol,afrimmlu_translate_wol,afrixnli_en_direct_wol,afrixnli_manual_direct_wol,afrixnli_manual_translate_wol,afrixnli_translate_wol,afrixnli_native_direct_wol,afrimgsm_direct_zul,afrimgsm_en_cot_zul,afrimgsm_translate_direct_zul,afrimmlu_direct_zul,afrimmlu_translate_zul,afrixnli_en_direct_zul,afrixnli_manual_direct_zul,afrixnli_manual_translate_zul,afrixnli_translate_zul,afrixnli_native_direct_zul"
EVAL_METRICS_EARLYSTOPPING="belebele_swh_Latn,belebele_yor_Latn,belebele_ibo_Latn,belebele_wol_Latn,belebele_zul_Latn,afrimgsm_direct_yor,afrimgsm_en_cot_yor,afrimgsm_translate_direct_yor,afrimmlu_direct_yor,afrixnli_en_direct_yor,afrixnli_manual_direct_yor,afrixnli_manual_translate_yor,afrixnli_native_direct_yor,afrixnli_translate_yor,afrimgsm_direct_ibo,afrimgsm_en_cot_ibo,afrimgsm_translate_direct_ibo,afrimmlu_direct_ibo,afrimmlu_translate_ibo,afrixnli_en_direct_ibo,afrixnli_manual_direct_ibo,afrixnli_manual_translate_ibo,afrixnli_translate_ibo,afrixnli_native_direct_ibo,afrimgsm_direct_wol,afrimgsm_en_cot_wol,afrimgsm_translate_direct_wol,afrimmlu_direct_wol,afrimmlu_translate_wol,afrixnli_en_direct_wol,afrixnli_manual_direct_wol,afrixnli_manual_translate_wol,afrixnli_translate_wol,afrixnli_native_direct_wol,afrimgsm_direct_zul,afrimgsm_en_cot_zul,afrimgsm_translate_direct_zul,afrimmlu_direct_zul,afrimmlu_translate_zul,afrixnli_en_direct_zul,afrixnli_manual_direct_zul,afrixnli_manual_translate_zul,afrixnli_translate_zul,afrixnli_native_direct_zul"
EARLY_STOPPING_PATIENCE=3
RESUME_FROM_CHECKPOINT=False

EVAL_BATCH_SIZE=8
EVAL_LIMIT=10
EVAL_CUDA_DEVICES="1"
EVAL_LOG_SAMPLES=true
EVAL_WANDB_PROJECT="gemma3-4b-pt-niger-congo-lanauge-adapter"


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
