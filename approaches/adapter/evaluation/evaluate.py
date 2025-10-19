import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import get_dataset_config_names, load_dataset


checkpoint_path = "/data/joel/phi2-lora-multilingual/checkpoint-7000/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

base_model_name_or_path = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()


def generate_answer(model, tokenizer, context, question, options):
    prompt = f"{context}\n\nQuestion: {question}\nOptions:\n"
    for i, opt in enumerate(options):
        prompt += f"{chr(65+i)}. {opt}\n"
    prompt += "\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer[-1].upper() if answer else "?"

print("Selecting langues/ Fetching language list from Belebele...")
#all_languages = get_dataset_config_names("facebook/belebele")
all_languages = ['deu_Latn', 'spa_Latn', 'fra_Latn', 'ita_Latn', 'por_Latn', 'pol_Latn', 'nld_Latn', 'ces_Latn', 'fas_Arab', 'ron_Latn', 'ukr_Cyrl', 'nob_Latn', 'ell_Grek', 'swe_Latn', 'dan_Latn']


results = {}

for lang in all_languages:
    print(f"\nEvaluating language: {lang}...")
    try:
        dataset = load_dataset("facebook/belebele", lang)["test"]
    except Exception as e:
        print(f"skip {lang} due to error: {e}")
        continue
    dataset = dataset.select(range(min(100, len(dataset))))

    correct = 0
    total = 0

    for example in dataset:
        context = example["flores_passage"]
        question = example["question"]
        options = [example[f"mc_answer{i}"] for i in range(1, 5)]
        gold = chr(64 + int(example["correct_answer_num"]))


        pred = generate_answer(model, tokenizer, context, question, options)
        if pred == gold:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    results[lang] = acc
    print(f"Accuracy on {lang}: {acc:.2%}")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

print("\nSummary of accuracies for eahc language")
for lang, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{lang:12s}: {acc:.2%}")
