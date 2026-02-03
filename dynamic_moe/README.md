# HTYLLM-PG

Multilingual Mixture-of-Experts Language Model Training and Evaluation.

## Quick Start

```bash
pip install -e .
```

## Reproducing Results

### 1. Download Data

Sample ~1TB of multilingual text from FineWeb subsets:

```bash
sbatch htyllm_pg/slurm/sample_fineweb.sh
```

This script:
- Creates a dataset inventory (`dataset_inventory.json`)
- Calculates sampling quotas for 128 languages
- Downloads and shards data to `/scratch/.../fineweb_samples/sharded_samples/`

### 2. Train Tokenizer

Train a BPE tokenizer (vocab size: 131,072) on the sampled data:

```bash
sbatch htyllm_pg/slurm/train_tokenizer.sh

# Or locally:
python htyllm_pg/train_tokenizer.py /path/to/fineweb_samples/ --vocab_size 131072
```

Outputs `tokenizer.json` and `tokenizer_config.json`.

### 3. Tokenize Data

Pack and tokenize documents into 2048-length sequences:

```bash
sbatch htyllm_pg/slurm/tokenize_multilingual.sh

# Or locally:
python -m htyllm_pg.tokenize_data \
    /path/to/sharded_samples \
    /path/to/tokenized_output \
    tokenizer.json \
    2048 \
    10000
```

### 4. Train Model

Launch 4-node distributed training:

```bash
sbatch htyllm_pg/slurm/train_multilingual_3_7b.sh
```

**Key hyperparameters** (in training script):
| Parameter | Value |
|-----------|-------|
| Nodes | 4 (16 H100 GPUs) |
| Batch size | 6 per GPU |
| Gradient accumulation | 16 |
| Learning rate | 1e-4 |
| Model dim | 3072 |
| Layers | 28 (24 MoE) |
| Experts | 8 |
| Expert parallelism | 8 |

Checkpoints are saved every 2000 steps to `--checkpoint-dir`.

### 5. Evaluate All Checkpoints

Run evaluation on all checkpoints using the Belebele and XNLI benchmarks:

```bash
python htyllm_pg/slurm/run_all_evals.py \
    --checkpoints-dir /path/to/checkpoints \
    --script-path htyllm_pg/slurm/convert_and_eval.sh
```

This script:
1. Finds all `step_*` checkpoint directories
2. Submits SLURM jobs to convert each to HuggingFace format
3. Runs `lm_eval` on 122 Belebele language tasks + XNLI
4. Logs results to Weights & Biases

**Dry run** (preview commands without executing):
```bash
python htyllm_pg/slurm/run_all_evals.py --checkpoints-dir /path/to/checkpoints --dry-run
```

## Manual Evaluation

For a single checkpoint:

```bash
# 1. Convert DeepSpeed → HuggingFace
deepspeed htyllm_pg/conversion_scripts/convert_ds_to_hf.py \
    --checkpoint_path checkpoints/step_10000 \
    --output_dir hf_models/step_10000 \
    --config_path htyllm_pg/conversion_scripts/config_3_7b.json

# 2. Run lm_eval
lm_eval --model hf \
    --model_args pretrained=hf_models/step_10000,trust_remote_code=True \
    --tasks belebele_eng_Latn,xnli \
    --batch_size auto
```

## Model Architecture

Build a custom MoE model:

```python
from htyllm_pg.model_builder import moe_builder

model = moe_builder(
    vocab_size=131072,
    max_seq_len=2048,
    dim=3072,
    depth=28,
    heads=24,
    mlp_dim=12288,
    moe_layers=list(range(4, 28)),  # MoE on layers 4-27
    num_experts=8,
    k=-1,                            # Dynamic routing
    gate_backward='ste',             # STE for binary gating
    ep_size=1
)
```

## Modifications to DynMoE

This implementation extends [DynMoE](https://arxiv.org/abs/2405.14297) with three key changes (see [`sharded_moe.py`](DeepSpeed/deepspeed/moe/sharded_moe.py)):

1. **Sigmoid-Derivative STE for Binary Gating**  
   DynMoE uses identity gradients (grad=1) for the binary expert selection. We use the sigmoid derivative `σ(1-σ)` in the backward pass, providing smoother gradient flow while keeping hard 0/1 outputs in forward.
   ```python
   gate_backward='ste'   # Ours: sigmoid derivative
   gate_backward='sign'  # Original DynMoE: identity
   ```

2. **Z-Loss for Router Stability**  
   Added `0.001 × logsumexp(pre_sigmoid)²` loss to prevent extreme router logits that cause unstable routing.

3. **Sparse Dispatch**  
   Memory-efficient implementation using `index_add` that returns sparse indices instead of dense `[T, E, C]` tensors, enabling scaling to longer context lengths and more experts.

## Files

| File | Description |
|------|-------------|
| `tokenizer.json` | Trained BPE tokenizer (131K vocab) |
| `ds_config.json` | DeepSpeed training config |
| `lm_eval_tasks.txt` | Evaluation tasks (122 Belebele + XNLI) |
| `htyllm_pg/conversion_scripts/config_3_7b.json` | Model architecture config |
