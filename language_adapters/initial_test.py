import json
from datasets import Dataset
from adapter_transformers import (
    AutoModelWithHeads,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

model = AutoModelWithHeads.from_pretrained("microsoft/deberta-v3-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

model.add_adapter("french")
model.train_adapter("french")
model.add_classification_head("french", num_labels=2)
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

french_ds = load_dataset_from_json("/data/joel/prepared/langauge_adapters/angeluriot_french_instruct_")

training_args = TrainingArguments(
    output_dir="./results-french",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs-french",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=french_ds,
)

trainer.train()
