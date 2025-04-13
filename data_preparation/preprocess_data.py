import os
import time
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./output"))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--tokenizer", default=os.getenv("TOKENIZER", "microsoft/phi-2"))
    parser.add_argument("--hf_cache_dir", default=os.getenv("HF_CACHE_DIR", "/data/hf_cache/"))
    parser.add_argument("--chunk_size", type=int, default=2048)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    proc_rank = int(os.getenv("PROC_RANK", "0"))
    total_procs = int(os.getenv("TOTAL_PROCS", "1"))
    os.environ["HF_HOME"] = args.hf_cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    for item in args.datasets:
        dataset_name, dataset_config, tokenizer_name = item.split(":")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        
        dataset = load_dataset(dataset_name, dataset_config, split="train", cache_dir=args.hf_cache_dir)
        subset = dataset.shard(num_shards=total_procs, index=proc_rank)
        print("data laoded, next create subsets")
        
        dataset_output_dir = os.path.join(args.output_dir, f"{dataset_name.replace('/', '_')}_{dataset_config}")
        os.makedirs(dataset_output_dir, exist_ok=True)
    
        all_tokens = []
        shard = []
        chunk_count = 0
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

        output_path = os.path.join(dataset_output_dir, f"shard_{proc_rank}.json")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(shard, f_out)

        elapsed = time.time() - start_time
        print(f"Process {proc_rank}/{total_procs}: {chunk_count} chunks saved in {elapsed:.2f}s. Output -> {args.output_dir}")

if __name__ == "__main__":
    main()