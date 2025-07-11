#!/bin/bash

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0

# ======= PARAMETERS =======
BASE_DIR="/data/joel/results_language_adapters/mistral7b/jav_Latn_sun_Latn/"
TOKENIZER_NAME="mistralai/Mistral-7B-v0.3"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
#TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
TASKS="belebele_sun_Latn,belebele_jav_Latn"
BATCH_SIZE="16"
LIMIT="None"
WANDB_PROJECT="mistral7b-jav_Latn_sun_Latn-adapter"
WANDB_GROUP="mistral7b-jav_Latn_sun_Latn-adapter"

LOG_FILE="$BASE_DIR/lm_harness_nohup_run_eval_jav_Latn_sun_Latn.log"

# ======= RUN SCRIPT =======
nohup python lm_harness_single.py \
  --base_dir "$BASE_DIR" \
  --tokenizer_name "$TOKENIZER_NAME" \
  --model_name "$MODEL_NAME" \
  --eval_tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" > "$LOG_FILE" 2>&1 &

sleep 2
echo "Training started with PID $!"
echo "Logging to: $LOG_FILE"
tail -f "$LOG_FILE"
