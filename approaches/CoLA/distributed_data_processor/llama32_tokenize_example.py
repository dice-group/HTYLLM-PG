#!/usr/bin/env python3
"""Tiny helper that tokenizes a sentence with the Llama 3.2 1B tokenizer."""
from transformers import AutoTokenizer

# "adapt that this is tokenized and printed /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples/dzo_Tibt/00000.jsonl.gz "

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    text = "Быстрая коричневая лиса прыгает через ленивую собаку"

    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Token count: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {tokens}")


if __name__ == "__main__":
    main()
