from transformers import BertForPreTraining, TrainingArguments, Trainer, BertTokenizer
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(torch.cuda.is_available())

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

class BinaryPretrainingDataset(Dataset):
    def __init__(self, bin_file: str, block_size: int):
        self.bin_file = bin_file
        self.block_size = block_size
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        input_ids = torch.tensor(self.data[i: i + self.block_size], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Mask 15% of tokens for MLM (BERT-style)
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
        input_ids[mask_arr] = tokenizer.mask_token_id

        # Dummy NSP label (assuming single sequences, not sentence pairs)
        next_sentence_label = torch.tensor(1, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "next_sentence_label": next_sentence_label
        }

# Load datasets
train_bin = "tokenized_data/train.bin"
val_bin = "tokenized_data/val.bin"
train_dataset = BinaryPretrainingDataset(train_bin, block_size=256)
val_dataset = BinaryPretrainingDataset(val_bin, block_size=256)

# Model and training setup
model = BertForPreTraining.from_pretrained('bert-base-multilingual-uncased')
if torch.cuda.is_available():
    model = model.to('cuda')

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    dataloader_num_workers=4,
    #strategy="ddp",  # Explicitly use DDP for multi-GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print(f"Model device: {next(model.parameters()).device}")
trainer.train()