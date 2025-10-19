import os
import sys
from pathlib import Path
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
from torch.profiler import profile, record_function, ProfilerActivity
from lm_harness_eval import LMEvalCallback
import glob

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

class CausalPretrainingDataset(Dataset):
    def __init__(self, bin_file: str, block_size: int, start_idx: int = 0, length: int = None):
        self.block_size = block_size
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.start_idx = start_idx
        self.length = length if length is not None else (len(self.data) - 1) // self.block_size - start_idx
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        i = actual_idx * self.block_size
        input_ids = torch.tensor(self.data[i: i + self.block_size], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone()
        }

def split_binary_file(bin_file: str, block_size: int = 256, split_ratio: float = 0.9):
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    total_len = (len(data) - 1) // block_size
    split_idx = int(total_len * split_ratio)
    
    train_data = CausalPretrainingDataset(bin_file, block_size, start_idx=0, length=split_idx)
    val_data = CausalPretrainingDataset(bin_file, block_size, start_idx=split_idx, length=total_len - split_idx)
    
    return train_data, val_data

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

# Suppress specific warning
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

# Ensure dataset path is provided and shared across all ranks
if len(sys.argv) < 2:
    raise ValueError("Usage: python mistral_finetune.py <path_to_bin_file>")

given_dataset = sys.argv[1]
if not Path(given_dataset).is_file() or not given_dataset.endswith('.bin'):
    raise FileNotFoundError(f"The given path ({given_dataset}) is not a valid .bin file.")

print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(f"Cuda available: {torch.cuda.is_available()}")

# Load datasets
print(f"Splitting the data located in {given_dataset} into train and val data.")
train_dataset, val_dataset = split_binary_file(given_dataset)
model_location = "./models" + "/mistral7b_fine_tuned_model_" + Path(given_dataset).stem

# Check for existing checkpoints
checkpoint_dir = model_location + '/checkpoints'
resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir)
if resume_from_checkpoint:
    print(f"Found existing checkpoint: {resume_from_checkpoint}")
    print("Training will resume from this checkpoint.")
else:
    print("No existing checkpoint found. Starting fresh training.")

model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    attn_implementation="eager",  # Use Flash Attention 2 for H100
    torch_dtype=torch.bfloat16,  # Set model dtype to bf16
)
model.config.pad_token_id = tokenizer.pad_token_id

if torch.cuda.is_available():
    model = model.to("cuda")

# Determine appropriate number of workers based on CPU count
#num_workers = min(os.cpu_count(), 8)  # Cap at 8 to avoid excessive overhead

training_args = TrainingArguments(
    output_dir=model_location + '/checkpoints',
    save_strategy="steps",
    save_steps=10000,
    logging_dir=model_location + '/logs',
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    num_train_epochs=1,
    weight_decay=0.01,
    bf16=True,  # Use bf16 for better numerical stability with large models
    dataloader_num_workers=16,  # Optimize for H100's high throughput
    ddp_find_unused_parameters=False,  # Set to False for better performance
    max_steps=100010,  
    remove_unused_columns=False,  # Important for custom datasets
    # Added for best model saving
    # eval_strategy="steps",
    # eval_steps=1000,
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    # save_total_limit=2,  # Keep best + latest checkpoint
)

print("Fine-tuning on the given dataset now...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[LMEvalCallback(
        tokenizer_name='mistralai/Mistral-7B-v0.3',
        eval_interval=10001,
        eval_tasks=["belebele_nya_Latn", "belebele_sna_Latn", "belebele_swh_Latn", "belebele_sun_Latn", "belebele_jav_Latn"],
        output_dir=os.path.join(model_location, "lm_eval"),
        tb_logdir=model_location + "/runs/lm_eval"
    )]
)

profile_dir = "profile_results/mistral7b_16-07-2025"
os.makedirs(profile_dir, exist_ok=True)

if is_main_process():
    print("Starting profiling...")

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #schedule=torch.profiler.schedule(wait=9000, warmup=10, active=20, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_flops=True
) as prof:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    

if is_main_process():
    print("Profiling complete.")
    print(prof.key_averages().table(sort_by="flops", row_limit=10))

# Save the model
model.save_pretrained(model_location)
tokenizer.save_pretrained(model_location)

if is_main_process():
    print(f"Saved the model and tokenizer to {model_location}.")
    if resume_from_checkpoint:
        print(f"Training resumed from: {resume_from_checkpoint}")
    print("Best model (lowest eval loss) has been saved.")