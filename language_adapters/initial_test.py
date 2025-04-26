import json
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from adapters import AutoAdapterModel
from transformers import DataCollatorForLanguageModeling

model = AutoAdapterModel.from_pretrained("microsoft/deberta-v3-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15,)

model.add_adapter("french", config="pfeiffer")
model.add_masked_lm_head("french")
model.train_adapter("french")
model.set_active_adapters("french")

# model.add_adapter("german")
# model.train_adapter("german")
# model.add_classification_head("german", num_labels=2)

def load_dataset_from_json(path):
    data = []
    for shard in ["shard_0.json", "shard_1.json"]:
        with open(f"{path}/{shard}", "r", encoding="utf-8") as f:
            data.extend(json.load(f))
    return Dataset.from_list(data)

def preprocess(example):
    return {
        "input_ids": example["tokens"],
        "attention_mask": [1] * len(example["tokens"]),
    }

french_ds = load_dataset_from_json("/data/joel/prepared/langauge_adapters/angeluriot_french_instruct_")
french_ds = french_ds.map(preprocess)

training_args = TrainingArguments(
    output_dir="./results-french",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs-french",
    save_strategy="epoch",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=french_ds,
    data_collator=data_collator,
)

trainer.train()
