#!/usr/bin/env python3
import torch
from htyllm_pg.model_builder import moe_builder

# ---------------------------
# Configuration
# ---------------------------
CHECKPOINT_PATH = "hf_model/pytorch_model.bin"
OUTPUT_LOGITS_PATH = "hf_logits_verification.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # or torch.float16 if you want to test FP16
DUMMY_TOKENS = [[1, 2, 3, 4]]  # tiny input sequence for verification

# ---------------------------
# Build model (must match converted HF architecture)
# ---------------------------
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
    use_gradient_checkpointing=True
)

# ---------------------------
# Load HF checkpoint
# ---------------------------
state = torch.load(CHECKPOINT_PATH, map_location="cpu")
missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

print("Checkpoint loaded with strict=False")
print(f"Missing keys (expected in model but not in checkpoint): {missing_keys[:20]}")
print(f"Unexpected keys (in checkpoint but not in model): {unexpected_keys[:20]}")

# ---------------------------
# Patch MoE layers for CPU/GPU forward
# ---------------------------
for layer in model.transformer.layers:
    ff_module = layer[1]  # FFN/MoE module
    if hasattr(ff_module, "deepspeed_moe"):
        # passthrough forward to avoid DeepSpeed dependency
        ff_module.forward = lambda x, *args, **kwargs: (x, 0, None)


# ---------------------------
# Set model precision and device
# ---------------------------
model = model.to(DEVICE).to(DTYPE)
model.eval()  # no gradient

# ---------------------------
# Prepare dummy tokens
# ---------------------------
tokens = torch.tensor(DUMMY_TOKENS, device=DEVICE, dtype=torch.long)

# ---------------------------
# Forward pass
# ---------------------------
with torch.no_grad():
    logits, _ = model(tokens)

# ---------------------------
# Save logits for comparison
# ---------------------------
torch.save(logits.cpu(), OUTPUT_LOGITS_PATH)
print(f"Forward pass complete. Logits shape: {logits.shape}")
print(f"Logits saved to {OUTPUT_LOGITS_PATH}")