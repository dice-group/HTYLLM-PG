#!/bin/bash

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1

# ======= PARAMETERS =======
BASE_DIR="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/mistral7b/arabic_subset"
TOKENIZER_NAME="/upb/users/j/joeldag/profiles/unix/cs/results_language_adapters/mistral7b/arabic_subset/adapter"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
BATCH_SIZE="32"
LIMIT="50"
WANDB_PROJECT="mistral7b_arabic_adapter_evaluation"
WANDB_GROUP="htyllm_mistral_eval_arabic_qlora"

# ======= RUN SCRIPT =======
python lm_harness_single.py \
  --base_dir "$BASE_DIR" \
  --tokenizer_name "$TOKENIZER_NAME" \
  --model_name "$MODEL_NAME" \
  --eval_tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP"
