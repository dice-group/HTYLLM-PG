#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online

# === Config ===
#TASKS="hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,xnli,pawsx,flores,wmt16,lambada_multilingual,xquad"
#TASKS="belebele_apc_Arab,belebele_ary_Arab,belebele_arz_Arab,belebele_ars_Arab,belebele_heb_Hebr,include_base_44_hindi,belebele_hin_Deva,belebele_urd_Arab,include_base_44_urdu,belebele_mar_Deva,include_base_44_bengali,belebele_ben_Beng,belebele_pan_Guru,belebele_swh_Latn,belebele_yor_Latn,belebele_ibo_Latn,belebele_wol_Latn,belebele_zul_Latn,afrimgsm_direct_yor,afrimgsm_en_cot_yor,afrimgsm_translate_direct_yor,afrimmlu_direct_yor,afrixnli_en_direct_yor,afrixnli_manual_direct_yor,afrixnli_manual_translate_yor,afrixnli_native_direct_yor,afrixnli_translate_yor,afrimgsm_direct_ibo,afrimgsm_en_cot_ibo,afrimgsm_translate_direct_ibo,afrimmlu_direct_ibo,afrimmlu_translate_ibo,afrixnli_en_direct_ibo,afrixnli_manual_direct_ibo,afrixnli_manual_translate_ibo,afrixnli_translate_ibo,afrixnli_native_direct_ibo,afrimgsm_direct_wol,afrimgsm_en_cot_wol,afrimgsm_translate_direct_wol,afrimmlu_direct_wol,afrimmlu_translate_wol,afrixnli_en_direct_wol,afrixnli_manual_direct_wol,afrixnli_manual_translate_wol,afrixnli_translate_wol,afrixnli_native_direct_wol,afrimgsm_direct_zul,afrimgsm_en_cot_zul,afrimgsm_translate_direct_zul,afrimmlu_direct_zul,afrimmlu_translate_zul,afrixnli_en_direct_zul,afrixnli_manual_direct_zul,afrixnli_manual_translate_zul,afrixnli_translate_zul,afrixnli_native_direct_zul,include_base_44_vietnamese,belebele_vie_Latn,belebele_tha_Thai,belebele_jav_Latn,belebele_sun_Latn,belebele_khm_Khmr"
TASKS="belebele_tur_Latn,include_base_44_turkish,turkishmmlu,belebele"
LIMIT="None"
OUTPUT_DIR="/data/joel/results_language_adapters/base_gemma_3_4b_evaluation_turkish"
MODEL_NAME="google/gemma-3-4b-pt"
WANDB_PROJECT="htyllm_base_gemma_3_4b_eval_only_turkish"
WANDB_GROUP="base_gemma_3_4b_eval_all_eval_tasks_turkish"

mkdir -p "$OUTPUT_DIR"

# === Run ===
nohup python evaluate_base_model.py \
  --output_dir "$OUTPUT_DIR" \
  --tokenizer_name "$MODEL_NAME" \
  --model_name "$MODEL_NAME" \
  --tasks "$TASKS" \
  --batch_size 32 \
  --limit "$LIMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  > "$OUTPUT_DIR/belebele_eval.log" 2>&1 &


sleep 2  # small delay to let log file be created
tail -f "$OUTPUT_DIR/belebele_eval.log"