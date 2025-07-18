#!/bin/bash

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0

# ======= PARAMETERS =======
BASE_DIR="/data/joel/results_language_adapters/mistral7b/final_model_2/arabic"
TOKENIZER_NAME="mistralai/Mistral-7B-v0.3"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
#TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
TASKS="belebele_apc_Arab,belebele_arz_Arab,belebele_ars_Arab,belebele_arb_Arab,belebele_acm_Arab"
BATCH_SIZE="auto"
LIMIT="900"
WANDB_PROJECT="final_mistral7b_arabic_adapter_evaluation"


LOG_FILE="$BASE_DIR/${WANDB_PROJECT}.log"

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

PID=$!
echo "Started lm_harness_single.py with PID $PID. Logging to $LOG_FILE"
tail -f "$LOG_FILE"