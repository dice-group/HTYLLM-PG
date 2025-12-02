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
    Fetch list of parquet files using recursive search to handle 'train', 'test',
    or 'sample' folder variations automatically.
    """
    # Explicitly pass token if available to ensure worker processes have access
    token = os.getenv("HF_TOKEN")
    fs = HfFileSystem(token=token)
    files = []
    
    # --- CASE 1: FineWeb 1 (English Only) ---
    if "fineweb" in repo_id and "fineweb-2" not in repo_id:
        if lang == "eng_Latn":
            # 1. Try the 'sample' folder (Confirmed to exist in your logs)
            pattern = f"{repo_id}/sample/*.parquet"
            files = fs.glob(pattern)
            
            # 2. Fallback to main data if sample is empty
            if not files:
                print(f"[INFO] 'sample' folder empty for {repo_id}, switching to CC-MAIN")
                # Use recursive glob to catch any CC-MAIN folder
                pattern = f"{repo_id}/data/**/*.parquet"
                files = fs.glob(pattern)
        else:
            return []

    # --- CASE 2: FineWeb 2 (Multilingual) ---
    elif "fineweb-2" in repo_id:
        # Your logs showed: data/aai_Latn/test/000.parquet
        # We use a recursive glob (**) to find files regardless of being in 'train' or 'test'
        
        # Pattern: repo/data/lang/**/*.parquet
        pattern = f"{repo_id}/data/{lang}/**/*.parquet"
        files = fs.glob(pattern)
            
        # Filter out "removed" folders if they exist
        files = [f for f in files if "_removed" not in f]

    # --- CASE 3: Generic Fallback ---
    else:
        pattern = f"{repo_id}/{lang}/**/*.parquet"
        files = fs.glob(pattern)

    # --- CLEANUP ---
    # fs.glob returns full paths (e.g., "datasets/HuggingFaceFW/fineweb/...").
    # hf_hub_download expects paths relative to the repo root.
    clean_files = []
    
    for f in files:
        if repo_id in f:
            # We split by the repo_id. 
            # Example f: "datasets/HuggingFaceFW/fineweb/sample/00.parquet"
            # Split: ["datasets/", "sample/00.parquet"]
            # We take the last part.
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

    try:
        # Get file list
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
        
        # Create/Clear file
        with open(output_file, 'wb') as f:
            pass

        for file_path in files:
            if collected_bytes >= target_bytes:
                break

            # 1. Download File
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
            df['cumsum'] = df['bytes'].cumsum()
            
            remaining = target_bytes - collected_bytes
            
            # 4. Filter
            df_batch = df[df['cumsum'] <= remaining + (df['bytes'].mean() * 5)]
            if df_batch.empty and not df.empty and remaining > 0:
                df_batch = df.iloc[:1]

            # 5. Tokenizer Sampling
            if tokenizer_subset_file and not df_batch.empty:
                sample_size = min(len(df_batch), 100)
                tok_sample = df_batch.sample(sample_size)
                tokenizer_samples.extend(tok_sample['text'].tolist())

            # 6. Write to Disk
            if not df_batch.empty:
                # Use force_ascii=False to keep characters readable if possible
                json_block = df_batch[['text']].to_json(
                    orient='records', 
                    lines=True, 
                    force_ascii=False
                )
                
                with gzip.open(output_file, 'at', encoding='utf-8') as f_out:
                    f_out.write(json_block)
                    f_out.write('\n')

                batch_bytes = df_batch['bytes'].sum()
                collected_bytes += batch_bytes
                
            # 7. Cleanup
            os.remove(local_parquet)
            del df
            del df_batch
            
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

    except Exception as e:
        # Print detailed error but allow other languages to continue
        print(f"[FAILED] {lang}: {e}")
        # import traceback
        # traceback.print_exc()
        return lang, 0

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

    print("Merging tokenizer subsets...")
    with open(args.tokenizer_data, 'w', encoding='utf-8') as outfile:
        for fname in glob.glob(os.path.join(args.output, "**", "tokenizer_subset_*.jsonl"), recursive=True):
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    
    print(f"COMPLETED: {completed_bytes/1e9:.2f} GB total")

if __name__ == "__main__":
    main()