CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --stage sft --do_train \
  --model_name_or_path meta-llama/Llama-3.2-3B \
  --dataset gsm8k \
  --dataset_dir ./data \
  --template llama3 \
  --finetuning_type cola \
  --output_dir ./saves/smoke_test \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --num_A 1 \
  --num_B 1 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --use_cola_experts \
  --cola_num_experts 4 \
  --cola_top_k 2 \
  --cola_debug

: '
TODOs:
- use multilingual data: sample data as in cluster or run in cluster
- use multilingual tokenizer (maybe do one run with and without to compare complexity)
- preprocess data with llamafactory
- run longer MoE cola test
'