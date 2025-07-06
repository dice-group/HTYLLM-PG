from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("checkpoints/moe_tf_model", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("checkpoints/moe_tf_model")
    
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
output = pipe("Hallo", max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])
    