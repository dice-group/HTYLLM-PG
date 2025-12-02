import os
import json
import gc
import pandas as pd
import pyarrow.parquet as pq
import gzip
import random
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

from datasets import load_dataset


# Buffer this many lines before flushing to disk
WRITE_BUFFER_SIZE = 1000
# Approximate bytes per char for UTF-8 (slightly overestimates for safety)
BYTES_PER_CHAR_ESTIMATE = 1.5


def process_language(lang_row, output_dir, inventory, tokenizer_subset_file=None):
    lang = lang_row["lang"]
    source = lang_row["source"]
    target_bytes = lang_row["final_bytes"]
    
    if target_bytes <= 0:
        return lang, 0
    
    print(f"[START] {lang} (Target: {target_bytes/1e9:.3f} GB)")
    
    # Determine dataset parameters based on source
    if source == "fw2":
        repo_id = "HuggingFaceFW/fineweb-2"
        data_name = lang 
    else:
        repo_id = "HuggingFaceFW/fineweb"
        data_name = "default"

    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    output_file = os.path.join(lang_dir, "data.jsonl.gz")
    
    collected_bytes = 0
    tokenizer_samples = []
    write_buffer = []
    
    try:
        ds = load_dataset(
            repo_id, 
            name=data_name, 
            split="train", 
            streaming=True,
            trust_remote_code=True,
        )
        
        estimated_samples = int((target_bytes / 2000) * 1.2)
        ds_iter = iter(ds.take(estimated_samples))
        
        with gzip.open(output_file, "wt", encoding="utf-8", compresslevel=1) as f_out:
            
            for sample in ds_iter:
                if collected_bytes >= target_bytes:
                    break
                
                text = sample.get("text", "")
                if not text:
                    continue
                
                text_bytes = int(len(text) * BYTES_PER_CHAR_ESTIMATE)
                
                if collected_bytes + text_bytes > target_bytes:
                    break
                    
                json_line = json.dumps({"text": text}, ensure_ascii=False)
                write_buffer.append(json_line)
                
                if len(write_buffer) >= WRITE_BUFFER_SIZE:
                    f_out.write("\n".join(write_buffer) + "\n")
                    write_buffer = []
                
                # Reservoir sampling for tokenizer (memory efficient)
                if len(tokenizer_samples) < 1000:
                    tokenizer_samples.append(text)
                elif random.random() < 1000 / (collected_bytes / 2000):
                    # Replace random element (reservoir sampling)
                    idx = random.randint(0, 999)
                    tokenizer_samples[idx] = text
                
                collected_bytes += text_bytes
            
            # Flush remaining buffer
            if write_buffer:
                f_out.write("\n".join(write_buffer) + "\n")

    except Exception as e:
        print(f"[ERROR] {lang}: {e}")
        import traceback
        traceback.print_exc()
        return lang, collected_bytes

    # Save tokenizer subset
    if tokenizer_subset_file and tokenizer_samples:
        tok_file = os.path.join(lang_dir, f"tokenizer_subset_{lang}.jsonl")
        with open(tok_file, "w", encoding="utf-8") as f_tok:
            f_tok.write("\n".join(
                json.dumps({"text": t}, ensure_ascii=False) for t in tokenizer_samples
            ) + "\n")

    print(f"[DONE] {lang}: {collected_bytes/1e9:.3f} GB")
    return lang, collected_bytes


def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotas", type=str, default="sampling_quotas.csv")
    parser.add_argument("--inventory", type=str, default="dataset_inventory.json")
    parser.add_argument("--output", type=str, default="sampled_data")
    parser.add_argument("--tokenizer_data", type=str, default="tokenizer_training_data.jsonl")
    # OPTIMIZATION 6: Higher default workers - streaming uses minimal memory
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    df_quotas = pd.read_csv(args.quotas)
    with open(args.inventory, "r") as f:
        inventory = json.load(f)
    
    rows = [row for _, row in df_quotas.iterrows() if row["final_bytes"] > 0]
    
    # OPTIMIZATION 7: Sort by size DESCENDING (largest first)
    # - Big downloads take longest, start them early
    # - Small ones fill in gaps and finish while big ones run
    rows.sort(key=lambda r: r["final_bytes"], reverse=True)
    
    total_target = sum(r["final_bytes"] for r in rows)
    print(f"Starting processing of {len(rows)} languages...")
    print(f"Total target: {total_target/1e9:.2f} GB")
    print(f"Workers: {args.workers}")
    
    completed_bytes = 0
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_language, row, args.output, inventory, args.tokenizer_data): row["lang"]
            for row in rows
        }
        
        # OPTIMIZATION 8: Use as_completed for progress tracking
        for future in as_completed(futures):
            lang = futures[future]
            try:
                _, bytes_done = future.result()
                completed_bytes += bytes_done
                completed_count += 1
                progress = (completed_bytes / total_target) * 100
                print(f"[PROGRESS] {completed_count}/{len(rows)} languages, "
                      f"{completed_bytes/1e9:.2f}/{total_target/1e9:.2f} GB ({progress:.1f}%)")
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
