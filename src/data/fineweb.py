import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Configuration
local_dir = "fineweb_tokenized"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
shard_size = int(1e8)
nprocs = max(1, os.cpu_count()//2)  # Use half the CPU cores

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]  # Start with EOT token
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    assert tokens_np.max() < 2**16, "Token overflow"
    return tokens_np

def write_datafile(filename, tokens):
    np.save(filename, tokens)

if __name__ == "__main__":
    # Load all .jsonl.gz files recursively
    dataset = load_dataset(
        "json",
        data_files="fineweb2_subset/**/*.jsonl.gz",
        split="train",
        streaming=True  
    )


    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens = np.empty(shard_size, dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # Process documents in parallel
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            while len(tokens) > 0:
                remaining = shard_size - token_count
                
                if remaining > 0:
                    # Add to current shard
                    add_tokens = tokens[:remaining]
                    all_tokens[token_count:token_count+len(add_tokens)] = add_tokens
                    token_count += len(add_tokens)
                    tokens = tokens[remaining:]
                    
                    # Initialize progress bar if needed
                    if progress_bar is None:
                        progress_bar = tqdm(
                            total=shard_size,
                            unit="tokens",
                            desc=f"Shard {shard_index}"
                        )
                    progress_bar.update(len(add_tokens))
                
                if token_count >= shard_size:
                    # Save completed shard
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(
                        DATA_CACHE_DIR,
                        f"fineweb_{split}_{shard_index:06d}.npy"
                    )
                    write_datafile(filename, all_tokens)
                    
                    # Reset for next shard
                    shard_index += 1
                    token_count = 0
                    progress_bar = None
                    all_tokens = np.empty(shard_size, dtype=np.uint16)

        # Save final partial shard
        if token_count > 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR,
                f"fineweb_{split}_{shard_index:06d}.npy"
            )
            write_datafile(filename, all_tokens[:token_count])