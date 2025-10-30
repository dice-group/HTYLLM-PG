CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --stage sft --do_train \
  --model_name_or_path meta-llama/Llama-3.2-3B \
  --dataset gsm8k \
  --dataset_dir ./data \
  --template llama3 \
  --finetuning_type cola \
  --output_dir ./saves/smoke_test \
  --overwrite_output_dir \
  --max_samples 50 \
  --num_train_epochs 0.1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --num_A 1 \
  --num_B 1 \
  --lora_rank 4 \
  --lora_alpha 8

