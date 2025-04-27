import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Load model and tokenizer
model = AutoAdapterModel.from_pretrained("microsoft/deberta-v3-large")
tokenizer = AutoTokenizer.from_pretrained("./results-german")
model.load_adapter("./results-german/adapter", load_as="german")
model.set_active_adapters("german")
model.eval()

sentences = [
    "Das ist ein wunderschöner Tag.",
    "Ich liebe es, im [MASK] zu lesen.",
    "Deutschland ist ein [MASK] Land."
]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits

    if "[MASK]" in sentence:
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_logits = outputs[0, mask_token_index, :]

        top_k = 5
        probs = torch.softmax(mask_logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k, dim=-1)

        print(f"Input: {sentence}")
        for i in range(top_k):
            token = tokenizer.decode(top_indices[0, i])
            probability = top_probs[0, i].item()
            print(f"  Prediction {i+1}: {token} (prob: {probability:.4f})")
        print()
    else:
        print(f"Input: {sentence} -> Model processed successfully.\n")

prompt = "Wie geht es dir?"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs).logits

next_token_id = outputs[0, -1].argmax(dim=-1).item()
next_token = tokenizer.decode([next_token_id])

print(f"User: {prompt}")
print(f"Bot: {next_token}")
