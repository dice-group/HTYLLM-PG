# Model Training Pipeline

## Initial Setup

# Train tokenizer from raw data files
python tokenizer\train_tokenizer.py --files_glob "fineweb2_subset\**\*.jsonl.gz" --output_dir tokenizer

# Build initial model checkpoint 
python src/build.py --save_dir checkpoints/init

## Preprocess and tokenise data
# Data is saved to data\processed automatically
python src/preprocess.py --files "fineweb2_subset\**\*.jsonl.gz" --tokenizer tokenizer

## Training

### Local Testing
# Small-scale test run with minimal resources
python src/train.py `
    --dataset_dir data/processed `
    --model_path checkpoints/init `
    --tokenizer_path tokenizer `
    --output_dir checkpoints/local_test_run `
    --max_steps 10 `
    --batch_size 1 `
    --grad_accum 1 `
    --logging_steps 1

### Production Training

# Single GPU training
python src/train.py --dataset_dir data/processed --model_path checkpoints/init --tokenizer_path tokenizer --output_dir checkpoints/pretrain-run --epochs 3 --batch_size 4 --grad_accum 8

# Multi-GPU training with torchrun on a single machine (4 GPUs example)
torchrun --nproc_per_node=4 src/train.py \
    --dataset_dir data/processed \
    --model_path checkpoints/init \
    --tokenizer_path tokenizer \
    --output_dir checkpoints/distributed-run \
    --batch_size 4 \
    --grad_accum 4

# For distributed training across multiple nodes with SLURM, use train.sh
# bash train.sh


## Training Parameters Guide
# These formulas help determine appropriate training configuration
effective_batch = batch_size * grad_accum * #gpus
tokens_per_step = effective_batch * seq_length
max_steps ≈ total_tokens / tokens_per_step

# Example calculation
Tokens/step = 64 × 1024 ≈ 65,536 tokens