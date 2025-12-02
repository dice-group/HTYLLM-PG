import gzip
import os
import json
import glob
import random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Approximate bytes per char for UTF-8
BYTES_PER_CHAR_ESTIMATE = 1.5

def get_repo_files(repo_id, lang, source):
    """
    Fetch list of parquet files with specific logic for FineWeb structure.
    """
    fs = HfFileSystem()
    files = []
    
    # --- CASE 1: FineWeb 1 (English Only) ---
    if "fineweb" in repo_id and "fineweb-2" not in repo_id:
        if lang == "eng_Latn":
            # STRATEGY: Use the "sample-10BT" folder. 
            # It is a high-quality subset sitting at the root, perfect for sampling.
            pattern = f"{repo_id}/sample-10BT/*.parquet"
            files = fs.glob(pattern)
            
            # Fallback: If sample-10BT is missing, grab from a random dump
            if not files:
                print(f"[INFO] sample-10BT not found for {repo_id}")
        else:
            # FW1 is English only
            return []

    # --- CASE 2: FineWeb 2 (Multilingual) ---
    elif "fineweb-2" in repo_id:
        # 1. Try finding 'train' split specifically (preferred)
        train_pattern = f"{repo_id}/data/{lang}/train/*.parquet"
        files = fs.glob(train_pattern)
        
        # 2. If no train folder, try recursive search inside the lang folder
        if not files:
            deep_pattern = f"{repo_id}/data/{lang}/**/*.parquet"
            files = fs.glob(deep_pattern)
            
        # 3. Filter out "removed" folders if recursive grabbed them
        files = [f for f in files if "_removed" not in f]

    # --- CASE 3: Generic Fallback ---
    else:
        pattern = f"{repo_id}/{lang}/*.parquet"
        files = fs.glob(pattern)

    # --- CLEANUP ---
    # fs.glob returns full paths (e.g., "datasets/HuggingFaceFW/fineweb/...").
    # hf_hub_download expects paths relative to the repo root.
    clean_files = []
    
    for f in files:
        if repo_id in f:
            # Split by repo_id and take the second part to get relative path
            # e.g. "datasets/HuggingFaceFW/fw/data/x.parquet" -> "data/x.parquet"
            clean_path = f.split(repo_id + "/")[-1]
            clean_files.append(clean_path)
    
    return clean_files

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
        print(f"[SKIP] No files found for {lang} in {repo_id}")
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
                force_download=False 
            )

            # 2. Load into Pandas
            df = pd.read_parquet(local_parquet, columns=["text"])
            
            # 3. Calculate sizes
            df['bytes'] = df['text'].str.len() * BYTES_PER_CHAR_ESTIMATE
            
            # 4. Cumulative sum to find cutoff
            df['cumsum'] = df['bytes'].cumsum()
            remaining = target_bytes - collected_bytes
            
            # Filter df to what we need
            df_batch = df[df['cumsum'] <= remaining + (df['bytes'].mean() * 5)] # buffer
            
            # If batch is empty but we need data, take at least one if it fits roughly
            if df_batch.empty and not df.empty and remaining > 0:
                df_batch = df.iloc[:1]

            # 5. Reservoir Sampling for Tokenizer
            if tokenizer_subset_file and not df_batch.empty:
                # Sample up to 100 lines per file
                sample_size = min(len(df_batch), 100)
                tok_sample = df_batch.sample(sample_size)
                tokenizer_samples.extend(tok_sample['text'].tolist())

            # 6. Write to Disk
            if not df_batch.empty:
                # Convert to JSON string in one go (fast C backend)
                # force_ascii=False prevents escaping unicode characters
                json_block = df_batch[['text']].to_json(
                    orient='records', 
                    lines=True, 
                    force_ascii=False
                )
                
                # Append to compressed file
                with gzip.open(output_file, 'at', encoding='utf-8') as f_out:
                    f_out.write(json_block)
                    f_out.write('\n') # Ensure newline after the block

                batch_bytes = df_batch['bytes'].sum()
                collected_bytes += batch_bytes
                
                print(f"[{lang}] +{batch_bytes/1e6:.1f} MB | Total: {collected_bytes/1e9:.2f} GB")

            # 7. Cleanup
            os.remove(local_parquet)
            del df
            del df_batch
            del json_block

    except Exception as e:
        print(f"[ERROR] {lang}: {e}")
        import traceback
        traceback.print_exc()

    # Save tokenizer subset
    if tokenizer_subset_file and tokenizer_samples:
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotas", type=str, default="sampling_quotas.csv")
    parser.add_argument("--inventory", type=str, default="dataset_inventory.json")
    parser.add_argument("--output", type=str, default="sampled_data")
    parser.add_argument("--tokenizer_data", type=str, default="tokenizer_training_data.jsonl")
    parser.add_argument("--workers", type=int, default=16) 
    args = parser.parse_args()
    
    df_quotas = pd.read_csv(args.quotas)
    
    rows = [row for _, row in df_quotas.iterrows() if row["final_bytes"] > 0]
    rows.sort(key=lambda r: r["final_bytes"], reverse=True)
    
    total_target = sum(r["final_bytes"] for r in rows)
    print(f"Starting processing of {len(rows)} languages...")
    print(f"Total target: {total_target/1e9:.2f} GB")
    
    # Use ProcessPoolExecutor to bypass GIL
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