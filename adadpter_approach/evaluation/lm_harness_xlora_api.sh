lm_eval \
  --model local-completions \
  --tasks belebele_swh_Latn \
  --num_fewshot 0 \
  --output_path ./eval_results/belebele_swh.json \
  --model_args "model=default,base_url=http://localhost:1234/v1/completions,tokenizer=mistralai/Mistral-7B-v0.3,max_tokens=512,temperature=0,logprobs=5,batch_size=1,num_concurrent=1"
