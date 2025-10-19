
## Running the Pipeline

### 1. Train the Tokenizer

Run the tokenizer training script with the correct data path:

```bash
./vm_tokenize.sh
```

This runs:
```bash
python tokenizer/train_tokenizer.py --files_glob "/data/fineweb2_subset/**/*.jsonl.gz" --output_dir tokenizer
```

### 2. Preprocess the Data

Preprocess the dataset with the trained tokenizer:

```bash
./vm_preprocess.sh
```

This runs:
```bash
python src/preprocess.py --files "/data/fineweb2_subset/**/*.jsonl.gz" --tokenizer tokenizer --num_proc 32 --out_dir data/processed
```

### 3. Build the Initial Model

Create the initial Mixtral MoE model:

```bash
./vm_build.sh
```

This runs:
```bash
python src/build.py
```

### 4. Train the Model

Train the model using both GPUs with DeepSpeed:

```bash
./vm_train.sh
```

This runs:
```bash
torchrun --nproc_per_node=2 src/train.py [options]
```

The training script uses:
- Both H100 GPUs (CUDA_VISIBLE_DEVICES=0,1)
- Batch size of 8 per GPU
- Gradient accumulation of 16 steps
- DeepSpeed ZeRO-3 optimization
- LM-evaluation-harness for model evaluation

## Notes

- The data is expected to be in `/data/fineweb2_subset/`
- Checkpoints will be saved to `checkpoints/pretrain_run_vm/`
- Evaluation metrics will be logged and available in TensorBoard
- htop for cpu usage