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
  - `'ste'`: Straight-through estimator for binary gating (our approach)
  - `'sing'`: DynMoE gradient computation method

