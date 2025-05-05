import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer
import gzip
import json

# data_folder = "/scratch/hpc-prf-merlin/htyllm-pg/data/fineweb2_subset"
data_folder = "src/data/fineweb2_subset"
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

num_proc = 8


def process(example):
    text = example["text"]
    tokens = tokenizer(text, truncation=True, padding=False, return_attention_mask=False)
    return {
        "ids": tokens["input_ids"],
        "len": len(tokens["input_ids"]),
    }


def main():
    # Define the paths to the processed Parquet files.
    # These files are now saved in a project-root directory "pre_processed_data"
    dataset_files = {}
    preproc_dir = Path("pre_processed_data")
    preproc_dir.mkdir(exist_ok=True)

    for lang_folder in os.listdir(data_folder):
        if lang_folder.endswith('.jsonl'):
            lang_path = Path(data_folder) / lang_folder
            if lang_path.is_dir():
                lang_key = lang_folder.replace(".jsonl", "")  # e.g., "adl_Latn"

                records = []
                for gz_file in lang_path.glob("*.jsonl.gz"):
                    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if "text" in data:
                                    records.append({"text": data["text"]})
                            except json.JSONDecodeError:
                                continue  # skip corrupted lines

                if records:
                    out_path = preproc_dir / f"{lang_key}.parquet"
                    pd.DataFrame(records).to_parquet(out_path, index=False)
                    dataset_files[lang_key] = str(out_path)

    ds_dict = load_dataset("parquet", data_files=dataset_files)

    # Load the processed dataset splits as a DatasetDict.
    # Here we construct a dict with keys corresponding to splits.
    dataset_splits = {split: ds for split, ds in ds_dict.items()}

    print("Tokenizing the dataset splits...")
    tokenized = {}
    for split, ds in dataset_splits.items():
        print(f"Tokenizing {split} split...")
        tokenized[split] = ds.map(
            process,
            remove_columns=["text"],
            desc=f"Tokenizing {split} split",
            num_proc=num_proc,
        )

    # Define the output directory for the tokenized binary files.
    output_dir = Path("tokenized_data")
    output_dir.mkdir(exist_ok=True, parents=True)

    for lang, ds in ds_dict.items():
        print(f"Tokenizing language: {lang}")

        tokenized = ds.map(
            lambda example: tokenizer(example["text"], truncation=True),
            remove_columns=["text"],
            desc=f"Tokenizing {lang}",
            num_proc=num_proc,
        )

        # Save output binary
        arr_len = np.sum([len(ids) for ids in tokenized["input_ids"]], dtype=np.uint64)
        print(f"Total tokens in '{lang}': {arr_len}")

        filename = output_dir / f"{lang}.bin"
        dtype = np.uint16
        arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))

        idx = 0
        shard_count = min(1024, len(tokenized))  # Use the smaller value between 1024 and dataset size
        if shard_count == 0:
            print(f"No data available for language '{lang}'. Skipping...")
            continue

        for batch_idx in tqdm(range(shard_count), desc=f"Writing {filename}"):
            batch = tokenized.shard(num_shards=shard_count, index=batch_idx, contiguous=True).with_format("numpy")
            if len(batch["input_ids"]) > 0:
                arr_batch = np.concatenate(batch["input_ids"])
                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
        arr.flush()
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
