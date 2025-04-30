import os
import multiprocessing as mp
import numpy as np
import tiktoken
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
from tqdm import tqdm
import glob

# ------------------------------------------
# SETTINGS
local_dir = "fineweb2_downloaded"
shard_output_dir = "fineweb2_tokenized"
shard_size = int(1e8)  # 100M tokens per shard

# create output directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), shard_output_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ------------------------------------------
# STEP 1: Download dataset
snapshot_download(
    "HuggingFaceFW/fineweb-2",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["data/*/train/*"],
    ignore_patterns=["data/*_removed/*"],
    local_dir_use_symlinks=False,  # optional: make real copies instead of symlinks
)
print(f"Dataset downloaded to {local_dir}")

# ------------------------------------------
# STEP 2: Set up tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# ------------------------------------------
# STEP 3: Find all parquet files
parquet_files = sorted(glob.glob(os.path.join(local_dir, "data", "*", "train", "*.parquet")))

print(f"Found {len(parquet_files)} parquet files to process.")

# ------------------------------------------
# STEP 4: Tokenize and shard
nprocs = max(1, os.cpu_count() // 2)
shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

with mp.Pool(nprocs) as pool:
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file}...")
        table = pq.read_table(parquet_file, columns=["text"])
        texts = table.column("text").to_pylist()

        for tokens in pool.imap(tokenize, texts, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"fineweb2_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

# write remaining tokens
if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"fineweb2_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])
