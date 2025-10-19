import torch, torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, concatenate_datasets

TEACHER_ID = "meta-llama/Meta-Llama-3-8B"
STUDENT_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID, device_map="auto", torch_dtype=torch.bfloat16
).eval()

student = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID, torch_dtype=torch.bfloat16
)

tok = AutoTokenizer.from_pretrained(STUDENT_ID, use_fast=False)
tok.pad_token = tok.eos_token

valid_pairs = ["en-es", "en-fr", "en-zh", "en-ru"]
dataset_list = []

for pair in valid_pairs:
    print(f"Loading pair: {pair}")
    ds = load_dataset("opus100", pair, split="train[:2%]")
    src_lang, tgt_lang = pair.split("-")
    ds = ds.map(lambda x: {
        "source": x["translation"][src_lang],
        "target": x["translation"][tgt_lang]
    }, remove_columns=["translation"])

    dataset_list.append(ds)

multi_lang_ds = concatenate_datasets(dataset_list)
temperature = 2.0
kd_loss = nn.KLDivLoss(reduction="batchmean")

def collate(batch):
    src_texts = [ex["source"] for ex in batch]
    inputs = tok(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        teacher_logits = teacher(**inputs).logits / temperature

    inputs["teacher_logits"] = teacher_logits
    return inputs

# Custom Trainer
class KDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_logits = inputs.pop("teacher_logits")
        outputs = model(**inputs)
        s_logits = outputs.logits / temperature

        loss = kd_loss(
            nn.functional.log_softmax(s_logits, dim=-1),
            nn.functional.softmax(teacher_logits, dim=-1)
        ) * temperature**2

        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir="llama_multilingual_student",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    num_train_epochs=1,
    bf16=True,
    report_to="tensorboard",
    save_strategy="epoch",
    remove_unused_columns=False
)

trainer = KDTrainer(model=student, args=args, train_dataset=multi_lang_ds, data_collator=collate)
trainer.train()
