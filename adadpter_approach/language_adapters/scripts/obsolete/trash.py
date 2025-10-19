from peft import PeftModel, PeftConfig
import torch
from safetensors.torch import load_file

adapter_path = "/data/joel/bloom560m-belebele-languages/checkpoint-1000/"

# Load only the adapter weights
adapter_state = load_file(f"{adapter_path}/adapter_model.safetensors")
total_params = sum(p.numel() for p in adapter_state.values())

print(f"LoRA adapter has {total_params:,} trainable parameters")
