#!/usr/bin/env python3
import torch

# ---------------------------
# Paths to the logits files
# ---------------------------
HF_LOGITS_PATH = "/scratch/hpc-prf-merlin/martin/HTYLLM-PG/hf_logits_verification.pt"
DS_LOGITS_PATH = "/scratch/hpc-prf-merlin/martin/HTYLLM-PG/ds_logits_verification.pt"

# ---------------------------
# Load logits
# ---------------------------
hf_logits = torch.load(HF_LOGITS_PATH)
ds_logits = torch.load(DS_LOGITS_PATH)

# ---------------------------
# Compare shapes first
# ---------------------------
if hf_logits.shape != ds_logits.shape:
    print("❌ Logits shapes do not match!")
    print(f"HF shape: {hf_logits.shape}, DS shape: {ds_logits.shape}")
else:
    print(f"✅ Logits shapes match: {hf_logits.shape}")

# ---------------------------
# Compare values
# ---------------------------
# torch.allclose allows you to check numerical closeness with a tolerance
if torch.allclose(hf_logits, ds_logits, rtol=1e-5, atol=1e-6):
    print("✅ Logits match numerically within tolerance")
else:
    # Compute max difference for inspection
    diff = torch.abs(hf_logits - ds_logits)
    print("❌ Logits differ!")
    print(f"Max absolute difference: {diff.max().item()}")
    print(f"Mean absolute difference: {diff.mean().item()}")
