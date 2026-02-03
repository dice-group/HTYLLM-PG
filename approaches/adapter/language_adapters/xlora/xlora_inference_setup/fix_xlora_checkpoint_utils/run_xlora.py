from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel

base_id   = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "/data/joel/results_language_adapters/xlora/mistral7b/depth_1_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn/checkpoint-50/"

cfg = AutoConfig.from_pretrained(base_id)
cfg.use_cache = False

base  = AutoModelForCausalLM.from_pretrained(
            base_id, config=cfg,
            torch_dtype="auto", device_map="auto")

model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
model.eval()
print("Model loaded successfully with XLORA adapters.")