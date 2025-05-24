import os
import sys
from pathlib import Path
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import warnings
from lm_harness_eval import LMEvalCallback
from datasets import load_dataset


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
            "labels": input_ids.clone()
        }


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


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


# Suppress specific warning
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

# Ensure dataset path is provided and shared across all ranks
if len(sys.argv) < 2:
    raise ValueError("Usage: python bloom.py <path_to_bin_file>")
given_dataset = sys.argv[1]

if is_main_process():
    if not Path(given_dataset).is_file() or not given_dataset.endswith('.bin'):
        raise FileNotFoundError(f"The given path ({given_dataset}) is not a valid .bin file.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Cuda available: {torch.cuda.is_available()}")

# Load datasets in all processes
if is_main_process():
    print(f"Splitting the data located in {given_dataset} into train and val data.")
train_dataset, val_dataset = split_binary_file(given_dataset)

model_location = "./models" + "/bloom_fine_tuned_model_" + Path(given_dataset).stem

model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if torch.cuda.is_available():
    model = model.to("cuda")

training_args = TrainingArguments(
    output_dir=model_location + '/checkpoints',
    save_strategy="steps",
    save_steps=499,
    logging_dir=model_location + '/logs',
    logging_steps=20,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    dataloader_num_workers=28,
    ddp_find_unused_parameters=False,
)

if is_main_process():
    print("Fine-tuning on the given dataset now...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[LMEvalCallback(
        tokenizer_name=model_name,
        eval_interval=500,
        eval_tasks=[
            "hellaswag", "belebele", "mgsm_direct_bn", "mgsm_direct_ca", "mgsm_direct_de",
            "mgsm_direct_en", "mgsm_direct_es", "mgsm_direct_fr", "mgsm_direct_ja", "mgsm_direct_ru",
            "mgsm_direct_sw", "mgsm_direct_te", "mgsm_direct_th", "mgsm_direct_zh", "mela"
        ],
        output_dir=os.path.join(model_location, "lm_eval"),
        tb_logdir=model_location + "/runs/lm_eval"
    )]
)

trainer.train()
model.save_pretrained(model_location)

if is_main_process():
    print(f"Saved the model to {model_location}.")

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
