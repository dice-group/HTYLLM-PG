#!/bin/bash

lm_eval \
  --model hf \
  --model_args "pretrained=mistralai/Mistral-7B-v0.3,peft=/data/joel/results_language_adapters/xlora/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian/checkpoint-2000,dtype=bfloat16,device_map=auto,trust_remote_code=true,use_cache=False," \
  --tasks arc_easy,arc_challenge,hellaswag,lambada_openai \
  --batch_size 8 \
  --device cuda:0 \
  --output_path /data/joel/results_language_adapters/xlora/mistral7b/swh_Latn_sna_Latn_nya_Latn_south_asian/checkpoint-2000/eval_results
