import os
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--hf_cache_dir", default="/data/hf_cache/")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache_dir

    for item in args.datasets:
        dataset_name, dataset_config = item.split(":")
        print(f"Start to download {dataset_name} ({dataset_config}) to {args.hf_cache_dir}: \n")
        load_dataset(dataset_name, dataset_config, split="train", cache_dir=args.hf_cache_dir)
    print("All datasets downloaded.")

if __name__ == "__main__":
    main()
