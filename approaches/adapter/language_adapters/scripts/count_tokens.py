import argparse
from datasets import load_from_disk

def count_tokens(example):
    """" counting tokens, only countin input_ids ensuret hat we really only consider data which is latere used for training """
    return {"num_tokens": len(example["input_ids"])}

def main(args):
    print(f"load date from dir:  {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print("example entry for debugging:", dataset[0])

    print(f"Count tokens with {args.num_proc} processes")
    token_counts = dataset.map(count_tokens, num_proc=args.num_proc, desc="Counting tokens")
    total_tokens = sum(token_counts["num_tokens"])
    print(f"Total tokens: {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the tokenized dataset directory.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to use for token counting.")
    args = parser.parse_args()
    main(args)
