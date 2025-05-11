### inspired by https://www.kdnuggets.com/implement-cross-lingual-transfer-learning-mbert-hugging-face-transformers
from pathlib import Path
import re
import os
import pandas as pd
from torch.utils.data import DataLoader
import glob
from transformers import BertForPreTraining, TrainingArguments, Trainer, BertTokenizer, DataCollatorForLanguageModeling
import numpy as np
import torch
#torch.backends.cuda.matmul.allow_tf32 = True

import torch.distributed as dist
from torch.utils.data import Dataset
import warnings
import csv
from lm_harness_eval import LMEvalCallback

# Suppress the specific warning
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

if is_main_process():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(torch.cuda.is_available())


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
tokenized_data = "tokenized_data"  # path to folder containing the tokenized data


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


def split_binary_file(bin_file: str, split_ratio: float = 0.9):
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    total_len = (len(data) - 1) // 256  # number of sequences
    split_idx = int(total_len * split_ratio)

    train_data = BinaryPretrainingDataset(bin_file, 256)
    val_data = BinaryPretrainingDataset(bin_file, 256)

    train_data.__len__ = lambda: split_idx
    val_data.__len__ = lambda: total_len - split_idx
    val_data.__getitem__ = lambda idx: BinaryPretrainingDataset(bin_file, 256).__getitem__(idx + split_idx)

    return train_data, val_data


# Load datasets
english_train = tokenized_data + "/english.bin"
train_dataset, val_dataset = split_binary_file(english_train)

output_dir = "./results"

# Model and training setup
model = BertForPreTraining.from_pretrained('bert-base-multilingual-uncased')
if torch.cuda.is_available():
    model = model.to('cuda')

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="steps",       # Save based on steps
    save_steps=200,
    logging_dir='./logs',
    logging_steps=20,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    dataloader_num_workers=28,  # set to 0 for windows compatibility
    ddp_find_unused_parameters=False,
)
if is_main_process():
    print("Fine-tuning on English dataset now...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[LMEvalCallback(
            tokenizer_name='bert-base-multilingual-uncased',
            eval_interval=500,
            eval_tasks=["hellaswag","belebele"],
            output_dir=os.path.join(output_dir, "lm_eval"),
            tb_logdir="./runs/lm_eval"
        )]
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
model.save_pretrained("./results/mbert_fine_tuned_model")

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()