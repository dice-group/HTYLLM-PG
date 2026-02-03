import os
import torch
import torch.distributed as dist
from transformers import BertForPreTraining, BertTokenizer
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset
import numpy as np
import glob
from pathlib import Path
import pandas as pd

# Tokenizer and device
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

class BinaryPretrainingDataset(Dataset):
    def __init__(self, bin_file: str, block_size: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        input_ids = torch.tensor(self.data[i: i + self.block_size], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
        input_ids[mask_arr] = tokenizer.mask_token_id

        # Set labels to -100 for non-masked tokens (ignored in loss/accuracy)
        labels[~mask_arr] = -100
        next_sentence_label = torch.tensor(1, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "next_sentence_label": next_sentence_label
        }

def setup_ddp():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def compute_accuracy_on_batch(batch, model):
    outputs = model(**batch)
    prediction_logits = outputs.prediction_logits  # shape: [batch_size, seq_len, vocab_size]
    predicted_ids = prediction_logits.argmax(dim=-1)  # shape: [batch_size, seq_len]

    labels = batch["labels"]
    mask = labels != -100  # Only evaluate accuracy on masked tokens
    correct = (predicted_ids == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    return accuracy

def evaluate():
    setup_ddp()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    model = BertForPreTraining.from_pretrained('bert-base-multilingual-uncased').to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    k = 1000
    block_size = 256
    tokenized_data = "tokenized_data"
    results = []

    for bin_file in glob.glob(os.path.join(tokenized_data, "*.bin")):
        lang = Path(bin_file).stem
        full_dataset = BinaryPretrainingDataset(bin_file, block_size)
        if len(full_dataset) == 0:
            continue

        indices = np.linspace(0, len(full_dataset) - 1, min(k, len(full_dataset)), dtype=int)
        subset = Subset(full_dataset, indices)
        sampler = DistributedSampler(subset, shuffle=False)
        loader = DataLoader(subset, batch_size=8, sampler=sampler)

        accuracies = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            acc = compute_accuracy_on_batch(batch, model)
            accuracies.append(acc)

        avg_accuracy = np.mean(accuracies)
        if rank == 0:
            print(f"{lang}: accuracy={avg_accuracy:.4f}")
            results.append({"language": lang, "accuracy": avg_accuracy})

    if rank == 0:
        df = pd.DataFrame(results)
        df.to_csv("before_fine_tune_zeroshot_eval_accuracy.csv", index=False)
        print("Saved evaluation results to zeroshot_eval_accuracy.csv")

    cleanup_ddp()

if __name__ == "__main__":
    evaluate()
