#!/bin/bash

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0
ulimit -v $((20 * 1024 * 1024))  # Limit virtual memory to 20GB
export OMP_NUM_THREADS=1      
export TORCHDYNAMO_DISABLE=1  

# ======= PARAMETERS =======
BASE_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/gemma3-4b-pt/base_gemma_3_4b_evaluation_all_eval_tasks"
TOKENIZER_NAME="google/gemma-3-4b-pt"
MODEL_NAME="google/gemma-3-4b-pt"
#TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
TASKS="arc_challenge_mt,arc_multilingual,belebele,xnli,flores,wmt16,lambada_multilingual,hellaswag,mmlu,xcopa,pawsx,xstorycloze,xwinograd"
BATCH_SIZE="2"
LIMIT="100"
WANDB_PROJECT="htyllm_base_gemma_3_4b_eval_all_eval_tasks"
WANDB_GROUP="htyllm_base_gemma_3_4b_eval_all_eval_tasks"
#CUDA_DEVICES="1"

LOG_FILE="$BASE_DIR/evaluation_all_datasets.log"

# ======= RUN SCRIPT =======
nohup python lm_harness_single_TODO_single_model.py \
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
