import json
import glob
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from adapters import AutoAdapterModel
from transformers import DataCollatorForLanguageModeling

model = AutoAdapterModel.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, pad_to_multiple_of=8,)

#model.add_adapter("french", config="pfeiffer")
#model.add_masked_lm_head("french")
#model.train_adapter("french")
#model.set_active_adapters("french")

model.add_adapter("german", config="pfeiffer")
model.add_masked_lm_head("german")
model.train_adapter("german")
model.set_active_adapters("german")

def load_dataset_from_json(path):
    data = []
    for shard_path in glob.glob(f"{path}/shard_*.json"):
        with open(shard_path, "r", encoding="utf-8") as f:
            data.extend(json.load(f))
    return Dataset.from_list(data)

def preprocess(example):
    return {
        "input_ids": example["tokens"],
        "attention_mask": [1] * len(example["tokens"]),
    }

german_ds = load_dataset_from_json("/data/joel/prepared/langauge_adapters/DiscoResearch_germanrag_")
german_ds = german_ds.map(preprocess, remove_columns=["tokens"])

training_args = TrainingArguments(
    output_dir="./results/german-phi2",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./results/german-phi2/logs",
    save_strategy="steps",
    save_steps=1250,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=german_ds,
    data_collator=data_collator,
)

trainer.train()
tokenizer.save_pretrained("./results/german-phi2")
model.save_adapter("./results/german-phi2/adapter", "german")
