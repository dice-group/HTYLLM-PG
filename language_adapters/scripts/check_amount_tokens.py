from datasets import load_from_disk

dataset = load_from_disk("/data/fineweb2_subset_belebele_tokenized_bloom-560m")

def count_tokens(example):
    return {"num_tokens": len(example["input_ids"])}

# Use multiprocessing
token_counts = dataset.map(count_tokens, num_proc=32)
total_tokens = sum(token_counts["num_tokens"])
print(f"Total tokens: {total_tokens}")
