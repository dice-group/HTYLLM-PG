import os
import gzip
import uuid
import argparse
from datasets import Dataset
from transformers import AutoTokenizer

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

def tokenize_fn(batch, tokenizer):
    tokenized = tokenizer(batch["text"], truncation=True, max_length=512)
    return tokenized

def main(args):
    print(f"tokenizer used for tokenization: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Read data from: {args.data_dir}")
    raw_dataset = Dataset.from_generator(
        lambda: read_gz_folder(args.data_dir),
        cache_dir=os.path.join(args.cache_dir, str(uuid.uuid4())) #ensures no cached data is tokenized, we dont want this when using this script multiple times with different finewb2 subsets
    )

    print("Start tokenizing")
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenize_fn(batch, tokenizer),
        batched=True,
        num_proc=args.num_proc
    )

    tokenized_dataset.save_to_disk(args.tokenized_data_dir)
    print(f"tokenized data saved to here: {args.tokenized_data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .gz files with text data.")
    parser.add_argument("--tokenized_data_dir", type=str, required=True, help="Directory to save the tokenized dataset.")
    parser.add_argument("--model_name", type=str, default="bigscience/bloom-560m", help="Pretrained tokenizer model name.")
    parser.add_argument("--cache_dir", type=str, default="/tmp", help="Cache directory for temporary dataset storage.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes for tokenization.")

    args = parser.parse_args()
    main(args)
