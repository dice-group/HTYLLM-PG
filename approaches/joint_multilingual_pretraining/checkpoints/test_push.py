from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/moe_tf_model")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/moe_tf_model", use_fast=False)

print("Pushing model to hub...")
model.push_to_hub("moe_tf_model")
print("Pushing tokenizer to hub...")
tokenizer.push_to_hub("moe_tf_model")
print("Done!")
