# HTYLLM-PG

## Install

```bash
pip install -e .
```

## Training

### Local Training

```bash
deepspeed htyllm_pg/train.py --deepspeed_config ds_config.json

# With custom hyperparameters
deepspeed htyllm_pg/train.py --deepspeed_config ds_config.json \
    --epochs 1 --batch-size 8 --lr 0.0001 --workers 8
```

### SLURM Multi-Node Training

```bash
sbatch htyllm_pg/slurm/train_test.sh
```

## Model Conversion

Convert DeepSpeed checkpoints to Hugging Face format for evaluation:

```bash
# Single checkpoint
deepspeed htyllm_pg/conversion_scripts/convert_ds_to_hf.py \
    --checkpoint_path checkpoints/step_1000 \
    --output_dir hf_models/step_1000 \
    --config_path htyllm_pg/conversion_scripts/config_3_7b.json

# Batch conversion (SLURM)
sbatch htyllm_pg/conversion_scripts/convert_all.sh
```

## Evaluation

Run evaluation on English tasks using `lm-evaluation-harness`.
Ensure you have converted your model first (see above).

```bash
# Quick test
lm_eval --model hf \
    --model_args pretrained=hf_models/step_1000,trust_remote_code=True,dtype=float16 \
    --tasks arc_easy --device cuda:0 --batch_size 4

# Batch evaluation (SLURM)
sbatch htyllm_pg/slurm/run_lm_eval_english.sh
```

## Model Builder Usage

Build a Mixture-of-Experts Transformer using `moe_builder`:

```python
from htyllm_pg.model_builder import moe_builder

model = moe_builder(
    vocab_size=32000,
    max_seq_len=128,
    dim=768,              # Model dimension
    depth=4,              # Number of transformer layers
    heads=4,              # Number of attention heads
    mlp_dim=512,          # Feed-forward hidden dimension
    moe_layers=[0, 3],    # Which layers use MoE (0-indexed)
    num_experts=4,        # Number of experts per MoE layer
    k=-1,                 # -1 = dynamic routing (DynMoE), >0 = top-k experts
    gate_backward='ste',  # 'ste' = our binary gating gradients, 'sing' = DynMoE approach
    ep_size=1             # Expert parallelism group size
)
```

### Key Parameters

- **`k=-1`**: Enables dynamic routing from DynMoE (adaptive expert selection)
- **`gate_backward`**: 
  - `'ste'`: Using sigmoid derivative for backprop in binary gating (our approach)
  - `'sing'`: DynMoE gradient computation method (uses gradient = 1) 

### Useful
- `tail -f slurm-xxxxxx.out | stdbuf -oL tr '\r' '\n'`

### Tokenizers

- `tokenizer.json` and `tokenizer_config.json` are required for evaluation. 
- `tokenizer_norm.json` applies `normalizers.Lowercase()` and `normalizers.StripAccents()`.
- `tokenizer.json` should be used for general purposes.

