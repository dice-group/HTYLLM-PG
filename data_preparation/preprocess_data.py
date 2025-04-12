import os
import time
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./output"))
    parser.add_argument("--dataset_name", default=os.getenv("DATASET_NAME", "allenai/c4"))
    parser.add_argument("--dataset_config", default=os.getenv("DATASET_CONFIG", "realnewslike"))
    parser.add_argument("--chunk_size", type=int, default=1024)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    proc_rank = int(os.getenv("PROC_RANK", "0"))
    total_procs = int(os.getenv("TOTAL_PROCS", "1"))

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
    print("data laoded, next create subsets")
    #subset = dataset.shuffle(seed=42).shard(num_shards=total_procs, index=proc_rank) #i assume shuffling is taking the concurrency right? so thats why it takes so long, TODO: find solution for later
    subset = dataset.shard(num_shards=total_procs, index=proc_rank)

    all_tokens = []
    shard = []
    chunk_count = 0
    shard_idx = 0
    start_time = time.time()

    for example in subset:
        text = example.get("text", "")
        if not text:
            continue
        #tokens = text.lower().strip().split()  # for debugging
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=args.chunk_size)
        if len(tokens) < 5:
            continue
        all_tokens.extend(tokens)

        while len(all_tokens) >= args.chunk_size:
            chunk = all_tokens[:args.chunk_size]
            all_tokens = all_tokens[args.chunk_size:]
            shard.append({"tokens": chunk})
            chunk_count += 1

            if len(shard) >= 10000:  #this is appropriate for c4 realnewslike TODO: calculate this, or use huggingface map function later
                output_path = os.path.join(args.output_dir, f"shard_{proc_rank}_{shard_idx}.json")
                with open(output_path, "w", encoding="utf-8") as f_out:
                    json.dump(shard, f_out)
                shard = []
                shard_idx += 1

    # Save remaining chunks
    if shard:
        output_path = os.path.join(args.output_dir, f"shard_{proc_rank}_{shard_idx}.json")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(shard, f_out)

    elapsed = time.time() - start_time
    print(f"Process {proc_rank}/{total_procs}: {chunk_count} chunks saved in {elapsed:.2f}s. Output -> {args.output_dir}")

if __name__ == "__main__":
    main()