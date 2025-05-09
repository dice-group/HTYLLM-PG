import os
import gzip
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

DATA_DIR = "/data/fineweb2_subset_belebele/"
TOKENIZED_DATA_DIR = "/data/fineweb2_subset_belebele_tokenized_bloom-560m/"
MODEL_NAME = "bigscience/bloom-560m"

def read_gz_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                path = os.path.join(root, file)
                with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and len(line) > 10:
                            yield {"text": line}

raw_dataset = Dataset.from_generator(lambda: read_gz_folder(DATA_DIR))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, num_proc=24)
tokenized_dataset.save_to_disk(TOKENIZED_DATA_DIR)
print(f"tokenized data saved to {TOKENIZED_DATA_DIR}")
