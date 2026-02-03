import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flops_profiler.profiler import get_model_profile
import json
import io
import sys


model_name = "google/gemma-3-4b-pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Create a dummy batch (1 × block_size)
block_size = 256
batch = torch.ones((1, block_size), dtype=torch.long, device=device)

# Redirect stdout to capture printed profiling details
temp_stdout = io.StringIO()
sys_stdout = sys.stdout
sys.stdout = temp_stdout

flops, macs, params = get_model_profile(
    model=model,
    kwargs={"input_ids": batch, "attention_mask": torch.ones_like(batch)},
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=3,
    warm_up=5,
    as_string=True,
)

# Restore original stdout
sys.stdout = sys_stdout

# Save the profiling report
profile_report_text = temp_stdout.getvalue()

# Save summary metrics and full report
profile_data = {
    "FLOPs": flops,
    "MACs": macs,
    "Parameters": params,
    "Full_Report": profile_report_text
}

with open("gemma_model_profiler_16-07-2025.json", "w") as f:
    json.dump(profile_data, f, indent=4)

print("Full model profiling data saved to gemma_model_profiler_16-07-2025.json")