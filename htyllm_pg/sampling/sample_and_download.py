import os
import json
import gc
import pandas as pd
import pyarrow.parquet as pq
import gzip
import random
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor

# Batch size for reading parquet row groups to avoid OOM
BATCH_SIZE = 10000

from datasets import load_dataset

def process_language(lang_row, output_dir, inventory, tokenizer_subset_file=None):
    lang = lang_row["lang"]
    source = lang_row["source"]
    target_bytes = lang_row["final_bytes"]
    
    if target_bytes <= 0:
        return
    
    print(f"Processing {lang} (Target: {target_bytes/1e6:.2f} MB)...")
    
    # Determine dataset parameters based on source
    if source == "fw2":
        repo_id = "HuggingFaceFW/fineweb-2"
        # FineWeb-2 usually requires a subset name, assuming 'lang' maps to config
        data_name = lang 
    else:
        repo_id = "HuggingFaceFW/fineweb"
        data_name = lang

    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    output_file = os.path.join(lang_dir, "data.jsonl.gz")
    
    collected_bytes = 0
    tokenizer_samples = []
    
    # STREAMING: Starts instantly, no waiting for full file downloads
    # buffer_size ensures we get some randomness without downloading everything
    try:
        ds = load_dataset(
            repo_id, 
            name=data_name, 
            split="train", 
            streaming=True
        ).shuffle(seed=42, buffer_size=10_000) 
        
        with gzip.open(output_file, "wt", encoding="utf-8") as f_out:
            for sample in ds:
                if collected_bytes >= target_bytes:
                    break
                
                text = sample.get("text", "")
                if not text:
                    continue
                    
                text_bytes = len(text.encode('utf-8'))
                
                # Check if this single row pushes us over significantly
                if collected_bytes + text_bytes > target_bytes:
                    break
                    
                # Write to main output
                json_line = json.dumps({"text": text}, ensure_ascii=False)
                f_out.write(json_line + "\n")
                
                # Tokenizer sampling
                if len(tokenizer_samples) < 1000 and random.random() < 0.1:
                    tokenizer_samples.append(text)
                
                collected_bytes += text_bytes

    except Exception as e:
        print(f"Error streaming {lang}: {e}")

    # Save tokenizer subset
    if tokenizer_subset_file and tokenizer_samples:
         tok_file = os.path.join(lang_dir, f"tokenizer_subset_{lang}.jsonl")
         with open(tok_file, "w", encoding="utf-8") as f_tok:
             for t in tokenizer_samples:
                 f_tok.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print(f"Finished {lang}: {collected_bytes/1e6:.2f} MB")

def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotas", type=str, default="sampling_quotas.csv")
    parser.add_argument("--inventory", type=str, default="dataset_inventory.json")
    parser.add_argument("--output", type=str, default="sampled_data")
    parser.add_argument("--tokenizer_data", type=str, default="tokenizer_training_data.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    df_quotas = pd.read_csv(args.quotas)
    with open(args.inventory, "r") as f:
        inventory = json.load(f)
        
    # Sort by size (smallest first) to clear small tasks quickly? 
    rows = [row for _, row in df_quotas.iterrows() if row["final_bytes"] > 0]
    
    print(f"Starting processing of {len(rows)} languages...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for row in rows:
            futures.append(
                executor.submit(process_language, row, args.output, inventory, args.tokenizer_data)
            )
            
        for future in futures:
            future.result()

    # Merge tokenizer subsets
    print("Merging tokenizer subsets...")
    with open(args.tokenizer_data, 'w', encoding='utf-8') as outfile:
        for fname in glob.glob(os.path.join(args.output, "**", "tokenizer_subset_*.jsonl"), recursive=True):
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

if __name__ == "__main__":
    main()
