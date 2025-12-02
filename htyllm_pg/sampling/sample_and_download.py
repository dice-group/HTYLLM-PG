import argparse
import gzip
import os
import json
import glob
import random
import time
import pyarrow.parquet as pq
import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download, login
from concurrent.futures import ThreadPoolExecutor, as_completed

BYTES_PER_CHAR_ESTIMATE = 1.5


def get_repo_files(repo_id, lang, token):
    """Fetch list of parquet files for a language from a HuggingFace dataset."""
    if token:
        try:
            login(token=token)
        except:
            pass

    fs = HfFileSystem(token=token)
    fs_prefix = f"datasets/{repo_id}"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if "fineweb-2" in repo_id:
                pattern = f"{fs_prefix}/data/{lang}/**/*.parquet"
                files = fs.glob(pattern)
                files = [f for f in files if "_removed" not in f]
            else:
                # FineWeb 1 (English only)
                if lang != "eng_Latn":
                    return []
                pattern = f"{fs_prefix}/sample/**/*.parquet"
                files = fs.glob(pattern)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"[WARN] Listing failed for {lang} (Attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] Could not list files for {lang} after {max_retries} attempts: {e}")
                return []

    return [f.split(repo_id + "/")[-1] for f in files if repo_id in f]


def process_language(lang_row, output_dir, token, tokenizer_subset_file=None):
    lang = lang_row["lang"]
    source = lang_row["source"]
    target_bytes = lang_row["final_bytes"]

    if target_bytes <= 0:
        return lang, 0

    repo_id = "HuggingFaceFW/fineweb-2" if source == "fw2" else "HuggingFaceFW/fineweb"

    try:
        files = get_repo_files(repo_id, lang, token)
        if not files:
            print(f"[SKIP] No files found for {lang} in {repo_id}")
            return lang, 0

        np.random.shuffle(files)

        lang_dir = os.path.join(output_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        output_file = os.path.join(lang_dir, "data.jsonl.gz")

        collected_bytes = 0
        tokenizer_samples = []

        # Clear file from any previous failed attempts
        open(output_file, 'wb').close()

        for file_path in files:
            if collected_bytes >= target_bytes:
                break

            try:
                local_parquet = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    token=token,
                    force_download=False
                )

                # Stream parquet in row groups to avoid loading entire file into memory
                parquet_file = pq.ParquetFile(local_parquet)
                
                for batch in parquet_file.iter_batches(batch_size=10000, columns=["text"]):
                    if collected_bytes >= target_bytes:
                        break
                    
                    texts = batch.column("text").to_pylist()
                    
                    # Calculate bytes and filter
                    remaining = target_bytes - collected_bytes
                    output_texts = []
                    batch_bytes = 0
                    
                    for text in texts:
                        text_bytes = len(text) * BYTES_PER_CHAR_ESTIMATE
                        if batch_bytes + text_bytes > remaining + 50000:  # small buffer
                            break
                        output_texts.append(text)
                        batch_bytes += text_bytes
                    
                    if not output_texts:
                        continue
                    
                    # Collect tokenizer samples
                    if tokenizer_subset_file:
                        sample_size = min(len(output_texts), 50)
                        tokenizer_samples.extend(random.sample(output_texts, sample_size))
                    
                    # Write as JSONL
                    with gzip.open(output_file, 'at', encoding='utf-8') as f_out:
                        for text in output_texts:
                            f_out.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                    
                    collected_bytes += batch_bytes

                os.remove(local_parquet)

            except Exception as e:
                print(f"[WARN] Failed to process {file_path} for {lang}: {e}")
                continue

        # Save tokenizer subset
        if tokenizer_subset_file and tokenizer_samples:
            if len(tokenizer_samples) > 2000:
                tokenizer_samples = random.sample(tokenizer_samples, 2000)

            tok_file = os.path.join(lang_dir, f"tokenizer_subset_{lang}.jsonl")
            with open(tok_file, "w", encoding="utf-8") as f_tok:
                for t in tokenizer_samples:
                    f_tok.write(json.dumps({"text": t}, ensure_ascii=False) + '\n')

        return lang, collected_bytes

    except Exception as e:
        print(f"[FAILED] {lang} crashed: {e}")
        return lang, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotas", type=str, default="sampling_quotas.csv")
    parser.add_argument("--output", type=str, default="sampled_data")
    parser.add_argument("--tokenizer_data", type=str, default="tokenizer_training_data.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[WARNING] HF_TOKEN not set, private datasets will fail.")
    else:
        print(f"[INFO] HF_TOKEN found ({hf_token[:5]}...)")

    import pandas as pd
    df_quotas = pd.read_csv(args.quotas)
    rows = [row for _, row in df_quotas.iterrows() if row["final_bytes"] > 0]
    rows.sort(key=lambda r: r["final_bytes"], reverse=True)

    total_target = sum(r["final_bytes"] for r in rows)
    print(f"Processing {len(rows)} languages, target: {total_target/1e9:.2f} GB")

    completed_bytes = 0
    completed_count = 0

    # ThreadPoolExecutor shares memory between threads (better for I/O-bound work)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_language, row, args.output, hf_token, args.tokenizer_data): row["lang"]
            for row in rows
        }

        for future in as_completed(futures):
            lang = futures[future]
            try:
                _, bytes_done = future.result()
                completed_bytes += bytes_done
                completed_count += 1
                progress = (completed_bytes / total_target) * 100
                print(f"[{completed_count}/{len(rows)}] {lang}: {completed_bytes/1e9:.2f} GB ({progress:.1f}%)")
            except Exception as e:
                print(f"[FAILED] {lang}: {e}")

    print("Merging tokenizer subsets...")
    with open(args.tokenizer_data, 'w', encoding='utf-8') as outfile:
        for fname in glob.glob(os.path.join(args.output, "**", "tokenizer_subset_*.jsonl"), recursive=True):
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

    print(f"DONE: {completed_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
