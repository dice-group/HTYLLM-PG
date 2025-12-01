import os
import json
import pandas as pd
import pyarrow.parquet as pq
import gzip
import random
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor

def process_language(lang_row, output_dir, inventory, tokenizer_subset_file=None):
    """
    Downloads parquet files for a language, samples rows to meet quota,
    writes to jsonl.gz, and saves a subset for tokenizer training.
    """
    lang = lang_row["lang"]
    source = lang_row["source"]
    target_bytes = lang_row["final_bytes"]
    
    if target_bytes <= 0:
        return
    
    print(f"Processing {lang} (Target: {target_bytes/1e6:.2f} MB)...")
    
    # Get file list for this language
    if source == "fw2":
        files = inventory["fineweb-2"][lang]["files"]
        repo_id = "HuggingFaceFW/fineweb-2"
    else:
        files = inventory["fineweb-1"]["files"]
        repo_id = "HuggingFaceFW/fineweb"
        
    # Shuffle files to avoid bias if we stop early
    random.shuffle(files)
    
    collected_bytes = 0
    
    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    
    output_file = os.path.join(lang_dir, "data.jsonl.gz")
    
    # Buffer for tokenizer training data (raw text)
    tokenizer_samples = []
    
    with gzip.open(output_file, "wt", encoding="utf-8") as f_out:
        for file_info in files:
            if collected_bytes >= target_bytes:
                break
                
            try:
                # Download file
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_info["path"],
                    repo_type="dataset",
                    force_download=False # Use cache
                )
                
                # Read parquet
                table = pq.read_table(local_path)
                df = table.to_pandas()
                
                # Shuffle rows
                df = df.sample(frac=1.0)
                
                for _, row in df.iterrows():
                    text = row["text"]
                    text_bytes = len(text.encode('utf-8'))
                    
                    if collected_bytes + text_bytes > target_bytes:
                        # Take one last one maybe? strict cap
                        break
                        
                    # Write to main output
                    json_line = json.dumps({"text": text}, ensure_ascii=False)
                    f_out.write(json_line + "\n")
                    
                    # Save for tokenizer (e.g., 1% probability or fixed count)
                    # We want a representative subset across all languages
                    # A simple heuristic: keep first N chars or random sample
                    if len(tokenizer_samples) < 1000 and random.random() < 0.1:
                         tokenizer_samples.append(text)
                    
                    collected_bytes += text_bytes
                    
            except Exception as e:
                print(f"Error processing {file_info['path']}: {e}")
                
    # Append tokenizer samples to a common file (thread-safe handling needed if parallel)
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
