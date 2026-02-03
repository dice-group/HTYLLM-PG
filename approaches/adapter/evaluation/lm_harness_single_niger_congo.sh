#!/bin/bash

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=1

# ======= PARAMETERS =======
BASE_DIR="/data/joel/results_language_adapters/mistral7b/final_model_2/niger_congo"
TOKENIZER_NAME="mistralai/Mistral-7B-v0.3"
MODEL_NAME="mistralai/Mistral-7B-v0.3"
#TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
TASKS="belebele_swh_Latn,belebele_zul_Latn,belebele_nso_Latn,belebele_tsn_Latn,belebele_sot_Latn,belebele_yor_Latn,belebele_ibo_Latn"
BATCH_SIZE="86"
LIMIT="150"
WANDB_PROJECT="final_mistral7b_niger_congo_adapter_evaluation_2"
WANDB_GROUP="htyllm_mistral_eval_niger_congo_qlora"

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
