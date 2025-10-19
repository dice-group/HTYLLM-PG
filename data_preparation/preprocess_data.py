import os
import time
import json
import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer

"""
Retrieve Text from differnt json structures in different datasets
"""
def extract_text(example):
    if "text" in example and example["text"]:
        return example["text"]
    if "conversation" in example:
        return " ".join(turn.get("text", "") for turn in example["conversation"] if turn.get("text"))
    if "content" in example and example["content"]:
        return example["content"]
    if "instruction" in example and example["instruction"]:
        return example["instruction"]
    if "contexts" in example and example["contexts"]:
        if isinstance(example["contexts"], list):
            return " ".join(example["contexts"])
        return example["contexts"]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./output"))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--tokenizer", default=os.getenv("TOKENIZER", "microsoft/phi-2"))
    parser.add_argument("--hf_cache_dir", default=os.getenv("HF_CACHE_DIR", "/data/hf_cache/"))
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    proc_rank = int(os.getenv("PROC_RANK", "0"))
    total_procs = int(os.getenv("TOTAL_PROCS", "1"))
    os.environ["HF_HOME"] = args.hf_cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    for item in args.datasets:
        dataset_name, dataset_config, tokenizer_name = item.split(":")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        logging.info(f"Loading dataset {dataset_name} config {dataset_config}")
        dataset = load_dataset(dataset_name, dataset_config, split="train", cache_dir=args.hf_cache_dir)
        logging.debug(f"Example entry: {dataset[0]}")
        logging.info(f"Dataset loaded. Number of examples: {len(dataset)}")

        subset = dataset.shard(num_shards=total_procs, index=proc_rank)
        
        dataset_output_dir = os.path.join(args.output_dir, f"{dataset_name.replace('/', '_')}_{dataset_config}")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        all_tokens = []
        shard = []
        chunk_count = 0
        start_time = time.time()

        for idx, example in enumerate(subset):
            text = extract_text(example)
            if not text:
                continue
            text = " ".join(text.strip().split())

            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=args.chunk_size)

            if len(tokens) < 5:
                continue
            #all_tokens.extend(tokens)
            shard.append({"tokens": tokens})

            while len(all_tokens) >= args.chunk_size:
                chunk = all_tokens[:args.chunk_size]
                all_tokens = all_tokens[args.chunk_size:]
                shard.append({"tokens": chunk})
                chunk_count += 1

            if idx % 10000 == 0:
                logging.info(f"Processed {idx} examples, {chunk_count} chunks ready so far.")

        output_path = os.path.join(dataset_output_dir, f"shard_{proc_rank}.json")
        logging.info(f"Saving {len(shard)} chunks to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(shard, f_out)
        elapsed = time.time() - start_time
        logging.info(f"Process {proc_rank}/{total_procs}: {chunk_count} chunks saved in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()