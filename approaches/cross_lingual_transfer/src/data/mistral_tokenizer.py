import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import gzip
import json

model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

num_proc = 48


def process(example):
    text = example["text"]
    tokens = tokenizer(text, padding=False, return_attention_mask=False)
    return {
        "ids": tokens["input_ids"],
        "len": len(tokens["input_ids"]),
    }


def main():
    if len(sys.argv) < 2:
        given_dataset = input("Please enter the path to the directory containing the jsonl files.\n")
        output_directory = input("Please enter the output directory path (or press Enter for default).\n")
        if not output_directory.strip():
            output_directory = None
    elif len(sys.argv) == 2:
        given_dataset = sys.argv[1]
        output_directory = None
    elif len(sys.argv) == 3:
        given_dataset = sys.argv[1]
        output_directory = sys.argv[2]
    else:
        print("Usage: python script.py <input_directory> [output_directory]")
        return

    dataset_dir = Path(given_dataset)
    if not dataset_dir.is_dir():
        print(f"The given path ({given_dataset}) is not a valid directory!")
        return

    # Find all .jsonl files and directories in the directory
    jsonl_items = list(dataset_dir.glob("*.jsonl"))
    if not jsonl_items:
        print(f"No .jsonl files or directories found in {given_dataset}")
        return

    print(f"Found {len(jsonl_items)} jsonl items: {[f.name for f in jsonl_items]}")

    # Define the paths to the processed Parquet files.
    # These files are now saved in a project-root directory "pre_processed_data"
    if output_directory:
        output_base = Path(output_directory)
        output_base.mkdir(exist_ok=True, parents=True)
        preproc_dir = output_base / "pre_processed_data"
    else:
        preproc_dir = dataset_dir / "pre_processed_data"
    preproc_dir.mkdir(exist_ok=True)

    # Combine all jsonl files into a single dataset
    all_records = []
    
    for jsonl_item in jsonl_items:
        print(f"Processing {jsonl_item.name}...")
        
        if jsonl_item.is_file():
            # Handle regular .jsonl files
            if jsonl_item.suffix == '.jsonl':
                with open(jsonl_item, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "text" in data:
                                all_records.append({"text": data["text"]})
                        except json.JSONDecodeError:
                            continue  # skip corrupted lines
            elif jsonl_item.name.endswith('.jsonl.gz'):
                # Handle .jsonl.gz files
                with gzip.open(jsonl_item, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "text" in data:
                                all_records.append({"text": data["text"]})
                        except json.JSONDecodeError:
                            continue  # skip corrupted lines
        elif jsonl_item.is_dir():
            # Handle directories that might contain .jsonl.gz files
            print(f"  {jsonl_item.name} is a directory, looking for .jsonl.gz files inside...")
            gz_files = list(jsonl_item.glob("*.jsonl.gz"))
            if gz_files:
                print(f"  Found {len(gz_files)} .jsonl.gz files in {jsonl_item.name}")
                for gz_file in gz_files:
                    print(f"    Processing {gz_file.name}...")
                    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if "text" in data:
                                    all_records.append({"text": data["text"]})
                            except json.JSONDecodeError:
                                continue  # skip corrupted lines
            else:
                print(f"  No .jsonl.gz files found in {jsonl_item.name}")
        else:
            print(f"  Skipping {jsonl_item.name} - not a file or directory")

    if not all_records:
        print("No valid records found in any of the jsonl files!")
        return

    print(f"Total records collected: {len(all_records)}")

    # Save combined data as parquet
    combined_parquet = preproc_dir / "combined_data.parquet"
    pd.DataFrame(all_records).to_parquet(combined_parquet, index=False)
    
    # Load as dataset
    dataset = Dataset.from_parquet(str(combined_parquet))
    
    print("Tokenizing the combined dataset...")
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing combined dataset",
        num_proc=num_proc,
    )

    # Define the output directory for the tokenized binary files.
    if output_directory:
        output_dir = Path(output_directory) / "tokenized_data"
    else:
        output_dir = dataset_dir / "tokenized_data"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Creating single combined binary file...")
    
    # Save output binary
    arr_len = np.sum([len(ids) for ids in tokenized["ids"]], dtype=np.uint64)
    print(f"Total tokens in combined dataset: {arr_len}")

    filename = output_dir / "combined_dataset.bin"
    dtype = np.uint16
    arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    shard_count = min(1024, len(tokenized))  # Use the smaller value between 1024 and dataset size
    if shard_count == 0:
        print("No data available in the combined dataset. Skipping...")
        return

    for batch_idx in tqdm(range(shard_count), desc=f"Writing {filename}"):
        batch = tokenized.shard(num_shards=shard_count, index=batch_idx, contiguous=True).with_format("numpy")
        if len(batch["ids"]) > 0:
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
    arr.flush()
    print(f"Saved combined dataset: {filename}")


if __name__ == "__main__":
    main()