import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

class CausalPretrainingDataset(Dataset):
    def __init__(self, bin_file: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    def __len__(self):
        return (len(self.data) - 1) // self.block_size
    def __getitem__(self, idx):
        i = idx * self.block_size
        input_ids = torch.tensor(self.data[i: i + self.block_size], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids
        }

def split_binary_file(bin_file: str, split_ratio: float = 0.9):
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    total_len = (len(data) - 1) // 256
    split_idx = int(total_len * split_ratio)
    train_data = CausalPretrainingDataset(bin_file, 256)
    val_data = CausalPretrainingDataset(bin_file, 256)
    train_data.__len__ = lambda: split_idx
    val_data.__len__ = lambda: total_len - split_idx
    val_data.__getitem__ = lambda idx: CausalPretrainingDataset(bin_file, 256).__getitem__(idx + split_idx)
    return train_data, val_data

def profile_training_manual_loop(model, train_dataset, tokenizer, profile_dir="profile_results"):
    """Profiles training for a few steps using a manual loop and torch.profiler."""
    os.makedirs(profile_dir, exist_ok=True)

    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            prof.step()
            if step >= 4:  # Only profile 5 steps
                break
    print(f"Profile saved to {profile_dir}.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

    if len(sys.argv) < 2:
        raise ValueError("Usage: python bloom.py <path_to_bin_file>")
    given_dataset = sys.argv[1]

    if is_main_process():
        if not Path(given_dataset).is_file() or not given_dataset.endswith('.bin'):
            raise FileNotFoundError(f"The given path ({given_dataset}) is not a valid .bin file.")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Cuda available: {torch.cuda.is_available()}")

    if is_main_process():
        print(f"Splitting the data located in {given_dataset} into train and val data.")

    train_dataset, val_dataset = split_binary_file(given_dataset)

    model_location = "./models" + "/bloom_fine_tuned_model_" + Path(given_dataset).stem
    model_name = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # --- Profile the training process using a few steps ---
    if is_main_process():
        print("Profiling training on the given dataset now...")
    profile_training_manual_loop(model, train_dataset, tokenizer)

    # --- Save the model ---
    model.save_pretrained("saved_model")
    print("Training complete and model saved.")
