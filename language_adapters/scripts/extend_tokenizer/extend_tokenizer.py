import argparse
import gzip
import json
from pathlib import Path
from transformers import AutoTokenizer

def read_jsonl_gz_texts(data_dir, max_lines=None):
    count = 0
    for lang_dir in Path(data_dir).iterdir():
        for gz_file in lang_dir.glob("*.jsonl.gz"):
            with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if max_lines and count >= max_lines:
                        return
                    obj = json.loads(line)
                    if 'text' in obj:
                        yield obj['text']
                        count += 1

def extract_new_tokens(corpus_iter, tokenizer, num_tokens):
    from collections import Counter
    import re

    token_counts = Counter()
    for text in corpus_iter:
        words = re.findall(r'\b\w+\b', text)
        token_counts.update(words)

    # Filter out tokens already in vocab
    existing_vocab = set(tokenizer.get_vocab().keys())
    print("Sample existing tokens:", list(existing_vocab)[:10])
    new_tokens = [tok for tok, _ in token_counts.most_common() if tok not in existing_vocab]
    print("Sample new tokens:", new_tokens[:10])
    return new_tokens[:num_tokens]

def extend_tokenizer(base_tokenizer_path, data_dir, output_dir, added_tokens, max_lines=None):
    print(f"Loading base tokenizer from {base_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)

    print(f"Reading text corpus from {data_dir}...")
    corpus_iter = list(read_jsonl_gz_texts(data_dir, max_lines=max_lines))

    print(f"Identifying top {added_tokens} new tokens...")
    new_tokens = extract_new_tokens(corpus_iter, tokenizer, added_tokens)

    print(f"Adding {len(new_tokens)} new tokens to tokenizer...")
    tokenizer.add_tokens(new_tokens)

    print(f"Saving updated tokenizer to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer extension complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--added_tokens", type=int, default=5000)
    parser.add_argument("--max_lines", type=int, default=None)

    args = parser.parse_args()

    extend_tokenizer(
        base_tokenizer_path=args.base_tokenizer,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        added_tokens=args.added_tokens,
        max_lines=args.max_lines
    )
