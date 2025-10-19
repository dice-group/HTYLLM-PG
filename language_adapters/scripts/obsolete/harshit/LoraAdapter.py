#!/usr/bin/env python
import os

# ————————————————
# Force the script to see ONLY GPU #1 as its CUDA device
# ————————————————
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # now physical GPU #1 appears as cuda:0

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# Verify that only one GPU is visible—and it’s your physical #1
print("Total visible CUDA devices:", torch.cuda.device_count())
print("Selected device name:", torch.cuda.get_device_name(0))

# Use the (now single) GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)  # ensure default CUDA device is set

# ————————————————
# Load and preprocess the dataset
# ————————————————
dataset = load_dataset("wiki40b", "de", split="train[:1%]")
# keep only the 'text' field and shuffle / select a small subset
dataset = dataset.map(lambda x: {"text": x["text"]}, remove_columns=dataset.column_names)
dataset = dataset.shuffle(seed=42).select(range(2000))

# ————————————————
# Load model & tokenizer
# ————————————————
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map=None,       # we handle placement ourselves
    load_in_8bit=False
).to(device)

# ————————————————
# Apply LoRA
# ————————————————
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ————————————————
# Tokenization
# ————————————————
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# ————————————————
# Data collator
# ————————————————
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)

# ————————————————
# Training arguments
# ————————————————
training_args = TrainingArguments(
    output_dir="./lora-falcon-de",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=3e-4,
    logging_steps=25,
    save_total_limit=2,
    save_steps=200,
    report_to="none",
    # no_cuda=False,  # default is fine, since CUDA_VISIBLE_DEVICES is set
)

# ————————————————
# Trainer & train
# ————————————————
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./lora-falcon-de")
