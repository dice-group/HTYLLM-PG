import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# -------------------------
# Configuration
# -------------------------
# Number of worker processes for the .map() call.
num_proc = 8

# Use tiktoken to get the GPT-2 encoding.
enc = tiktoken.get_encoding("gpt2")

# -------------------------
# Tokenization Function
# -------------------------
def process(example: dict) -> dict:
    """
    Tokenizes an example using tiktoken's GPT-2 BPE.
    
    - Encodes the given text without including any special tokens via encode_ordinary.
    - Appends the end-of-text (EOT) token (e.g. 50256 for GPT-2) after encoding.
    
    Args:
        example (dict): A dictionary containing a key "text".
    
    Returns:
        dict: A dictionary with 'ids': list of token IDs and 'len': length of the token list.
    """
    # Encode the text as a list of integers.
    ids = enc.encode_ordinary(example["text"])
    # Append the EOT token.
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

# -------------------------
# Main Tokenization Pipeline
# -------------------------
def main():
    # Define the paths to the processed Parquet files.
    # These files are now saved in a project-root directory "pre_processed_data"
    data_dir = Path("pre_processed_data")
    train_file = str(data_dir / "train_data.snap.parquet")
    val_file = str(data_dir / "eval_data.snap.parquet")

    # Load the processed dataset splits as a DatasetDict.
    # Here we construct a dict with keys corresponding to splits.
    dataset_files = {"train": train_file, "val": val_file}
    ds_dict = load_dataset("parquet", data_files=dataset_files)
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

    # Writing Binary Output Files
    for split, dset in tokenized.items():
        # Calculate the total number of tokens across the split.
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"Total tokens in '{split}' split: {arr_len}")

        # Construct the output binary file name in the tokenized_data directory.
        filename = output_dir / f"{split}.bin"
        # Use np.uint16 as dtype because the GPT-2 vocab size (50256) fits in 16 bits.
        dtype = np.uint16
        
        # Create a memmap array to hold all tokens.
        arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024  # Adjust as needed based on system resources

        idx = 0  # Track the current write position in the memmap.
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
            # Shard the dataset for faster batch processing.
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            if len(batch["ids"]) > 0:
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
        arr.flush()
        print(f"Saved binary file for split '{split}': {filename}")

    print("Tokenization complete.")

if __name__ == "__main__":
    main()
