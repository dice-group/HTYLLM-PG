import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", use_cache=False)
PeftModel.from_pretrained(base, "/data/joel/results_language_adapters/xlora/mistral7b/trash_peft_fails_test/xlora_adapter/")
print("Loaded model succesfully")