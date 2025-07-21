from transformers import BertForPreTraining, TrainingArguments, Trainer, BertTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import torch


class BinaryDataset(Dataset):
    """
    A PyTorch Dataset to read the tokenized binary file.

    The binary file is created with dtype=np.uint16 (which is sufficient for GPT-2 vocab).
    Each sample is a contiguous sequence of length block_size extracted from the file.
    The target is the input sequence shifted by one token.
    """

    def __init__(self, bin_file: str, block_size: int):
        self.bin_file = bin_file
        self.block_size = block_size
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        x = torch.tensor(self.data[i: i + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[i + 1: i + self.block_size + 1], dtype=torch.long)
        return x, y


def tokenize_function(examples):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    premise = [ex if isinstance(ex, str) else " ".join(ex) for ex in examples['premise']]
    hypothesis = [ex if isinstance(ex, str) else " ".join(ex) for ex in examples['hypothesis']]

    return tokenizer(premise, hypothesis, padding="max_length", truncation=True)


train_bin = "tokenized_data/train.bin"
val_bin = "tokenized_data/val.bin"

train_dataset = BinaryDataset(train_bin, block_size=256)
val_dataset = BinaryDataset(val_bin, block_size=256)

model = BertForPreTraining.from_pretrained('google-bert/bert-base-uncased')
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Evaluation (Idk if this works yet - cannot test on my device)

french_val_bin = "tokenized_data/french_val.bin"
french_val_dataset = BinaryDataset(french_val_bin, block_size=256)

results = trainer.evaluate(french_val_dataset)
print(results)
