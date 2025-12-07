import os
import torch
import torch.distributed as dist
import deepspeed
from htyllm_pg.model_builder import moe_builder

# ----------------------------
# Initialize single-process distributed (for MoE)
# ----------------------------
if not dist.is_initialized():
    dist.init_process_group(
        backend="gloo",  # "nccl" if you have GPU
        init_method="tcp://127.0.0.1:29500",
        rank=0,
        world_size=1
    )

# Init DeepSpeed communication
deepspeed.init_distributed(dist_backend='gloo')

# ----------------------------
# Configuration
# ----------------------------
CHECKPOINT_PATH = "/scratch/hpc-prf-merlin/martin/HTYLLM-PG/checkpoints/step_4100/mp_rank_00_model_states.pt"
OUTPUT_LOGITS_PATH = "ds_logits_verification.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # torch.float16 is optional
DUMMY_TOKENS = [[1, 2, 3, 4]]  # simple dummy input

# ----------------------------
# Build model (must match converted HF architecture)
# ----------------------------
model = moe_builder(
    vocab_size=262144,
    max_seq_len=2048,
    dim=2048,
    depth=24,
    heads=16,
    mlp_dim=8192,
    dim_head=128,
    moe_layers=[3, 7, 11, 15, 19, 23],
    num_experts=8,
    use_flash_attention=True,
    use_gradient_checkpointing=False  # safer for verification
)

# ----------------------------
# Load checkpoint
# ----------------------------
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

state = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state, strict=False)
print("Checkpoint loaded with strict=False")

# ----------------------------
# Set model precision and device
# ----------------------------
model = model.to(DEVICE).to(DTYPE)
model.eval()

# ----------------------------
# Prepare dummy tokens
# ----------------------------
tokens = torch.tensor(DUMMY_TOKENS, device=DEVICE, dtype=torch.long)

# MoE layers often expect a second input: a dummy mask for "used tokens"
# This ensures dispatching works
used_tokens = torch.ones_like(tokens, device=DEVICE, dtype=torch.bool)

# ----------------------------
# Forward pass
# ----------------------------
with torch.no_grad():
    try:
        logits, _ = model((tokens, used_tokens))  # pass as tuple for MoE
    except TypeError:
        # Some versions may ignore second argument
        logits, _ = model(tokens)

# ----------------------------
# Save logits
# ----------------------------
torch.save(logits.cpu(), OUTPUT_LOGITS_PATH)
print(f"Forward pass complete. Logits shape: {logits.shape}")
print(f"Logits saved to {OUTPUT_LOGITS_PATH}")