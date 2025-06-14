#!/bin/bash

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1

# ======= PARAMETERS =======
BASE_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/gemma3-4b-pt/bod_Tibt_kac_Latn_shn_Mymr"
TOKENIZER_NAME="google/gemma-3-4b-pt"
MODEL_NAME="google/gemma-3-4b-pt"
#TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
TASKS="belebele_kac_Latn,belebele_bod_Tibt,belebele_shn_Mymr"
BATCH_SIZE="16"
LIMIT="None"
WANDB_PROJECT="gemma3-4b-pt_bod_Tibt_kac_Latn_shn_Mymr_adapter_evaluation"
WANDB_GROUP="htyllm_gemma3-4b-pt_eval_pt_bod_Tibt_kac_Latn_shn_Mymr_qlora_gemma3-4b-pt"

LOG_FILE="$BASE_DIR/lm_harness_nohup_run_eval_bod_Tibt_kac_Latn_shn_Mymr.log"

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
