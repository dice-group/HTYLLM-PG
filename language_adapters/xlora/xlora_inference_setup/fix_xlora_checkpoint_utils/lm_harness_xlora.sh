lm_eval \
  --model hf \
  --model_args \
    "pretrained=mistralai/Mistral-7B-Instruct-v0.3,\
peft_adapter=/data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50,\
peft_type=xlora,\
trust_remote_code=True,\
use_cache=False,\
device_map=auto,\
dtype=float16" \
  --tasks belebele \
  --batch_size auto \
  --output_path xlora_mistral7b_results.json

PYTHONUNBUFFERED=1 lm_eval --model hf \
        --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.3,peft=/data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50,use_cache=False, \
        --tasks belebele_jav_Latn \
        --device cuda

## Das hier funktioniert finally, nachdem man den checkpoint in 12 schritten angepasst hat
lm_eval --model hf \
        --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.3,peft=/data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50,use_cache=False, \
        --tasks belebele_jav_Latn,belebele_sun_Latn \
        --device cuda
