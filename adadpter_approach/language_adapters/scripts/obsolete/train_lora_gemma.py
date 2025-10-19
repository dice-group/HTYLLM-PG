import os
import torch
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from torch.utils.data import DataLoader

# Load tokenized datasets
tokenized_path = "/data/finweb2_tokenized"
train_dataset = load_from_disk(os.path.join(tokenized_path, "train"))
eval_dataset = load_from_disk(os.path.join(tokenized_path, "test"))

loader = DataLoader(train_dataset, batch_size=1)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model and tokenizer
model_name = "google/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map={"": 0})

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="/data/joel/gemma-lora-multilingual",
    per_device_train_batch_size=14,
    per_device_eval_batch_size=14,
    #eval_strategy="steps",
    #eval_steps=100,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="steps",   
    logging_dir="/data/joel/gemma-lora-multilingual/logs",
    save_steps=20,
    save_safetensors=True,
    save_on_each_node=False,
    save_total_limit=2,
    fp16=True,
    report_to="tensorboard",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()
