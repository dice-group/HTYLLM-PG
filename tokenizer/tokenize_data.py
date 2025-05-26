import os
import sentencepiece as spm
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

# --- CONFIG ---
MODEL_PATH       = "tokenizer/sp_model.model"
INPUT_PATH       = "../data/corpus.txt"
OUT_DIR          = "tokenizer/shards"
TOKENS_PER_SHARD = 100_000_000
N_WORKERS        = min(cpu_count(), 32)  # match your SBATCH cpus-per-task

os.makedirs(OUT_DIR, exist_ok=True)

# --- LOAD MODEL ONCE ---
sp = spm.SentencePieceProcessor()
sp.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

def tokenize_line(line: str):
    return sp.encode(line.strip())  # encode as IDs instead of pieces

def main():
    shard_idx = 0
    token_count = 0
    current_shard = []

    with Pool(N_WORKERS) as pool, \
         open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         tqdm(total=13452947, desc="Lines", unit="line") as pbar:

        for ids in pool.imap(tokenize_line, fin, chunksize=1_000):
            line_tokens = len(ids)
            
            # Roll shard if this line would exceed the limit
            if token_count + line_tokens > TOKENS_PER_SHARD and current_shard:
                # Save current shard as numpy array
                np.save(f"{OUT_DIR}/shard_{shard_idx:04d}.npy", np.array(current_shard, dtype=np.int32))
                shard_idx += 1
                token_count = 0
                current_shard = []
            
            current_shard.extend(ids)  # add token IDs to current shard
            token_count += line_tokens
            pbar.update(1)

    # Save the last shard
    if current_shard:
        np.save(f"{OUT_DIR}/shard_{shard_idx:04d}.npy", np.array(current_shard, dtype=np.int32))

    print(f"Done: created {shard_idx+1} shard(s) in `{OUT_DIR}`")

if __name__ == "__main__":
    main()
