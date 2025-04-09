import os
import time
import json
import argparse
from datasets import load_dataset

def preprocess_text(text):
    return text.lower().strip().split()

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess C4 RealNewsLike dataset for LLM training."
    )
    parser.add_argument(
        "--output_dir",
        default=os.getenv("OUTPUT_DIR", "./output"),
    )
    parser.add_argument(
        "--dataset_name",
        default=os.getenv("DATASET_NAME", "allenai/c4"),
        help="Dataset to load (default: C4).",
    )
    parser.add_argument(
        "--dataset_config",
        default=os.getenv("DATASET_CONFIG", "realnewslike"),
        help="Dataset configuration (default: realnewslike).",
    )
    args = parser.parse_args()

    proc_rank = int(os.getenv("PROC_RANK", "0"))
    total_procs = int(os.getenv("TOTAL_PROCS", "1"))

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
    )

    subset = dataset.shuffle(seed=42).shard(num_shards=total_procs, index=proc_rank) #shuffle and shard

    #preprocess and save
    output_file = os.path.join(args.output_dir, f"processed_{proc_rank}.jsonl")
    line_count = 0
    token_count = 0
    start_time = time.time()

    with open(output_file, "w", encoding="utf-8") as f_out:
        for example in subset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = preprocess_text(text)
            f_out.write(json.dumps({"text": " ".join(tokens)}) + "\n")
            #f_out.write(json.dumps({"tokens": tokens}) + "\n")
            line_count += 1
            token_count += len(tokens)

    elapsed = time.time() - start_time
    print(
        f"Process {proc_rank}/{total_procs}: {line_count} lines, "
        f"{token_count} tokens processed in {elapsed:.2f}s. Output -> {output_file}"
    )

if __name__ == "__main__":
    main()
