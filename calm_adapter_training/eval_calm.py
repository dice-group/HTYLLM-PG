import torch
from transformers import AutoTokenizer
from model.calm import CALM, CALMConfig  # adjust this import path if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Define config ===
config = CALMConfig(
    anchor_model="google/gemma-7b",
    aug_model="google/gemma-2b",
    num_connections=2,
    num_heads=2,
)

# === 2. Load model ===
model = CALM.from_pretrained("/data/joel/calm-results/gemma2_7b", config=config)
model.to(device)
model.eval()

# === 3. Load tokenizer (from anchor) ===
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

# === 4. Prepare input ===
prompt = "The universe is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# === 5. Generate output ===
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
    )

# === 6. Decode and print ===
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nOutput:\n{output_text}")
