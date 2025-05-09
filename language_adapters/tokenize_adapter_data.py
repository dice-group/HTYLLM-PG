import os
import gzip
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

def tokenize_data(data_dir, model_name, tokenized_output_dir, num_processes):
    print(f"Tokenizing data from {data_dir} using tokenizer {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = Dataset.from_generator(lambda: read_gz_folder(data_dir))
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=512),
        batched=True, num_proc=num_processes
    )
    tokenized_dataset.save_to_disk(tokenized_output_dir)
    print(f"Tokenized data saved to {tokenized_output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--num_processes", required=True)

    args = parser.parse_args()

    tokenize_data(args.data_dir, args.model_name, args.tokenized_dir, int(args.num_processes))
