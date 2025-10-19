from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_paths = {
    "adapter_1": "/data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50/adapter_1/",
    "adapter_2": "/data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50/adapter_2/",
}

tokenizer = AutoTokenizer.from_pretrained(base_id)
cfg = AutoConfig.from_pretrained(base_id)
cfg.use_cache = False

base = AutoModelForCausalLM.from_pretrained(
    base_id,
    config=cfg,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    base,
    adapter_paths["adapter_1"],
    adapter_name="adapter_1",
    local_files_only=True,
)
model.load_adapter(
    adapter_paths["adapter_2"],
    adapter_name="adapter_2",
    local_files_only=True,
)

print("Loaded adapters:", list(model.peft_config.keys()))

text = "Translate to Swahili: Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

for adapter in ["adapter_1", "adapter_2"]:
    print(f"\nuse {adapter}")
    model.set_adapter(adapter)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output with {adapter}: {output_text}")
