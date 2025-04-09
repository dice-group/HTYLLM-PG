import os
import time
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess C4 RealNewsLike dataset into training chunks."
    )
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./output"))
    parser.add_argument("--dataset_name", default=os.getenv("DATASET_NAME", "allenai/c4"))
    parser.add_argument("--dataset_config", default=os.getenv("DATASET_CONFIG", "realnewslike"))
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size in tokens.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    proc_rank = int(os.getenv("PROC_RANK", "0"))
    total_procs = int(os.getenv("TOTAL_PROCS", "1"))

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=True)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
    subset = dataset.shuffle(seed=42).shard(num_shards=total_procs, index=proc_rank)

    all_tokens = []
    chunk_count = 0
    start_time = time.time()

    for example in subset:
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=args.chunk_size)
        if len(tokens) < 5:
            continue
        all_tokens.extend(tokens)

        # Save in fixed-size chunks
        while len(all_tokens) >= args.chunk_size:
            chunk = all_tokens[:args.chunk_size]
            all_tokens = all_tokens[args.chunk_size:]
            output_path = os.path.join(args.output_dir, f"chunk_{proc_rank}_{chunk_count}.json")
            with open(output_path, "w", encoding="utf-8") as f_out:
                json.dump({"tokens": chunk}, f_out)
            chunk_count += 1

    elapsed = time.time() - start_time
    print(f"Process {proc_rank}/{total_procs}: {chunk_count} chunks saved in {elapsed:.2f}s. Output -> {args.output_dir}")

if __name__ == "__main__":
    main()
