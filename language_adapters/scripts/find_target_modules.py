from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch
import torch.nn as nn

# Define find_all_linear_names manually
def find_all_linear_names(model):
    """
    Finds all unique final component names of torch.nn.Linear modules in the model.
    Example return: ['q_proj', 'v_proj', 'dense', ...]
    """
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if '.' in name:
                lora_module_names.add(name.split('.')[-1])
            else:
                lora_module_names.add(name)
    return list(lora_module_names)

model_name = "mistralai/Mistral-7B-v0.3"

# Load quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

linear_names = find_all_linear_names(model)
print("Recommended LoRA target_modules:")
print(linear_names)
