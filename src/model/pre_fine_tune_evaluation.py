from pathlib import Path
import os
import pandas as pd
from torch.utils.data import DataLoader
import glob
from transformers import BertForPreTraining, TrainingArguments, Trainer, BertTokenizer
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
import csv

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

# Evaluate on other languages here (with zero- / few-shot analysis)
model = BertForPreTraining.from_pretrained('bert-base-multilingual-uncased')
k = 1000 # change to higher k later
block_size = 256
tokenized_data = "tokenized_data"
model.eval()

results = []


def compute_loss_on_batch(batch):
    with torch.no_grad():
        outputs = model(**batch)
        return outputs.loss.item()


for bin_file in glob.glob(os.path.join(tokenized_data, "*.bin")):
    lang = Path(bin_file).stem

    # Load the full binary dataset
    full_dataset = BinaryPretrainingDataset(bin_file, block_size)

    # Sample k examples or fewer if not enough
    num_samples = min(k, len(full_dataset))
    if num_samples == 0:
        print(f"Skipping {lang} due to empty dataset.")
        continue

    # Sample k examples
    indices = np.linspace(0, len(full_dataset) - 1, num_samples, dtype=int)
    subset = torch.utils.data.Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=1)

    # Collect losses
    losses = []
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = compute_loss_on_batch(batch)
        losses.append(loss)

    avg_loss = np.mean(losses)
    results.append({"language": lang, "avg_loss": avg_loss})
    print(f"{lang}: loss={avg_loss:.4f}")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("before_fine_tune_zeroshot_eval_results.csv", index=False)
print("Saved evaluation results to before_fine_tune_fewshot_eval_results.csv")