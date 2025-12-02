import gzip
import os
import json
import glob
import random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from concurrent.futures import ProcessPoolExecutor, as_completed # Changed to Process
import time

# Approximate bytes per char for UTF-8
BYTES_PER_CHAR_ESTIMATE = 1.5

def get_repo_files(repo_id, lang, source):
    """
    Fetch list of parquet files for a specific language.
    """
    fs = HfFileSystem()
    try:
        if source == "fw2":
            # Fineweb-2 structure: data/{lang}/...
            pattern = f"{repo_id}/data/{lang}/*.parquet"
        else:
            pattern = f"{repo_id}/{lang}/*.parquet"
            # Fallback if the pattern is just root
            if not fs.glob(pattern):
                pattern = f"{repo_id}/data/*.parquet"

        files = fs.glob(pattern)
        # Clean paths to be relative to repo
        files = [f.replace(f"{repo_id}/", "") for f in files]
        return files
    except Exception as e:
        print(f"[WARN] Could not list files for {lang}: {e}")
        return []

def process_language_vectorized(lang_row, output_dir, tokenizer_subset_file=None):
    lang = lang_row["lang"]
    source = lang_row["source"]
    target_bytes = lang_row["final_bytes"]
    
    if target_bytes <= 0:
        return lang, 0

    # Setup Repo ID
    if source == "fw2":
        repo_id = "HuggingFaceFW/fineweb-2"
    else:
        repo_id = "HuggingFaceFW/fineweb"

    # Get file list and shuffle
    files = get_repo_files(repo_id, lang, source)
    if not files:
        # Fallback for subsets that might be defined differently in FW1
        # If FW1 uses config names rather than folders, this needs specific mapping
        # But assuming file path structure here:
        print(f"[SKIP] No files found for {lang}")
        return lang, 0
        
    np.random.shuffle(files)

    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    output_file = os.path.join(lang_dir, "data.jsonl.gz")

    collected_bytes = 0
    tokenizer_samples = []
    
    
    try:
        # Create/Clear file
        with open(output_file, 'wb') as f:
            pass

        for file_path in files:
            if collected_bytes >= target_bytes:
                break

            # 1. Download File (Fastest way using hf_transfer)
            local_parquet = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                force_download=False # Use cache if exists, but we usually delete
            )

            # 2. Load into Pandas (Vectorized - Super Fast)
            df = pd.read_parquet(local_parquet, columns=["text"])
            
            # 3. Calculate sizes
            # Vectorized length calculation
            df['bytes'] = df['text'].str.len() * BYTES_PER_CHAR_ESTIMATE
            
            # 4. Cumulative sum to find cutoff
            df['cumsum'] = df['bytes'].cumsum()
            
            # Check how much we need
            remaining = target_bytes - collected_bytes
            
            # Filter df to what we need
            df_batch = df[df['cumsum'] <= remaining + (df['bytes'].mean() * 2)] # slight buffer
            
            # If batch is empty but we need data, take at least one if it fits roughly
            if df_batch.empty and not df.empty and remaining > 0:
                df_batch = df.iloc[:1]

            # 5. Reservoir Sampling for Tokenizer (Vectorized-ish)
            # Take a random 0.1% sample or up to 100 rows per file for tokenizer
            if tokenizer_subset_file:
                tok_sample = df_batch.sample(min(len(df_batch), 100))
                tokenizer_samples.extend(tok_sample['text'].tolist())

            # 6. Write to Disk (Vectorized)
            # appending to jsonl.gz
            if not df_batch.empty:
                df_batch[['text']].to_json(
                    output_file, 
                    orient='records', 
                    lines=True, 
                    compression={'method': 'gzip', 'compresslevel': 1},
                    mode='a' # Append mode requires pandas >= 1.4ish logic or manual handling
                    # Pandas to_json append with compression is sometimes flaky.
                    # safer: write to string, then append to gzip file
                )
                
                # Manual append to ensure safety with GZIP
                with gzip.open(output_file, 'at', encoding='utf-8') as f_out:
                    for txt in df_batch['text']:
                        f_out.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

                batch_bytes = df_batch['bytes'].sum()
                collected_bytes += batch_bytes

            # 7. Cleanup to save scratch space
            os.remove(local_parquet)
            del df
            del df_batch
            
            print(f"[{lang}] Downloaded {file_path}, extracted {batch_bytes/1e6:.1f} MB. Total: {collected_bytes/1e9:.2f} GB")

    except Exception as e:
        print(f"[ERROR] {lang}: {e}")
        import traceback
        traceback.print_exc()

    # Save tokenizer subset
    if tokenizer_subset_file and tokenizer_samples:
        # Limit tokenizer samples to avoid memory explosion
        if len(tokenizer_samples) > 2000:
            tokenizer_samples = random.sample(tokenizer_samples, 2000)
            
        tok_file = os.path.join(lang_dir, f"tokenizer_subset_{lang}.jsonl")
        with open(tok_file, "w", encoding="utf-8") as f_tok:
            f_tok.write("\n".join(
                json.dumps({"text": t}, ensure_ascii=False) for t in tokenizer_samples
            ) + "\n")

    return lang, collected_bytes

def main():
    import argparse
    import gzip # Ensure gzip is imported
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotas", type=str, default="sampling_quotas.csv")
    parser.add_argument("--inventory", type=str, default="dataset_inventory.json")
    parser.add_argument("--output", type=str, default="sampled_data")
    parser.add_argument("--tokenizer_data", type=str, default="tokenizer_training_data.jsonl")
    parser.add_argument("--workers", type=int, default=16) # Match CPU count
    args = parser.parse_args()
    
    df_quotas = pd.read_csv(args.quotas)
    
    rows = [row for _, row in df_quotas.iterrows() if row["final_bytes"] > 0]
    # Sort largest to smallest
    rows.sort(key=lambda r: r["final_bytes"], reverse=True)
    
    total_target = sum(r["final_bytes"] for r in rows)
    print(f"Starting processing of {len(rows)} languages...")
    print(f"Total target: {total_target/1e9:.2f} GB")
    
    # CRITICAL CHANGE: ProcessPoolExecutor
    # This creates separate Python processes, bypassing the GIL.
    # Each process runs on a separate core.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_language_vectorized, row, args.output, args.tokenizer_data): row["lang"]
            for row in rows
        }
        
        completed_bytes = 0
        completed_count = 0
        
        for future in as_completed(futures):
            lang = futures[future]
            try:
                _, bytes_done = future.result()
                completed_bytes += bytes_done
                completed_count += 1
                progress = (completed_bytes / total_target) * 100
                print(f"[PROGRESS] {completed_count}/{len(rows)} | {lang} done | "
                      f"{completed_bytes/1e9:.2f} GB ({progress:.1f}%)")
            except Exception as e:
                print(f"[FAILED] {lang}: {e}")

    # Merge tokenizer subsets
    print("Merging tokenizer subsets...")
    with open(args.tokenizer_data, 'w', encoding='utf-8') as outfile:
        for fname in glob.glob(os.path.join(args.output, "**", "tokenizer_subset_*.jsonl"), recursive=True):
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    
    print(f"COMPLETED: {completed_bytes/1e9:.2f} GB total")

if __name__ == "__main__":
    main()