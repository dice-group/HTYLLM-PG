python tokenizer\train_tokenizer.py --files_glob "fineweb2_subset\**\*.jsonl.gz" --output_dir tokenizer

python src/build.py --save_dir checkpoints/init

### local testing:

python src/train.py `
    --model_path checkpoints/init `
    --tokenizer_path tokenizer `
    --data_glob "fineweb2_subset/**/*.jsonl.gz" `
    --output_dir checkpoints/local_test_run `
    --max_steps 10 `
    --batch_size 1 `
    --grad_accum 1 `
    --logging_steps 1


effective_batch = batch_size * grad_accum * #gpus
tokens_per_step  = effective_batch * seq_length
max_steps       ≈ total_tokens / tokens_per_step


Tokens/step = 64 × 1024 ≈ 65 536 tokens

we need max steps still!