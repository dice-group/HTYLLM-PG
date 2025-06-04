import argparse
import gzip
import json
from pathlib import Path
from transformers import AutoTokenizer

def read_jsonl_gz_texts(data_dir):
    for lang_dir in Path(data_dir).iterdir():
        for gz_file in lang_dir.glob("*.jsonl.gz"):
            with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    if 'text' in obj:
                        yield obj['text']
                        
def extend_tokenizer(base_tokenizer_path, data_dir, output_dir, added_tokens, max_lines=None):
    print(f"Loading base tokenizer from {base_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)

    print(f"Loading text data from {data_dir}...")
    corpus_iter = read_jsonl_gz_texts(data_dir)

    print(f"Training new tokenizer with +{added_tokens} tokens...")
    new_vocab_size = tokenizer.vocab_size + added_tokens
    new_tokenizer = tokenizer.train_new_from_iterator(corpus_iter, vocab_size=new_vocab_size)

    print(f"Saving new tokenizer to {output_dir}...")
    new_tokenizer.save_pretrained(output_dir)
    print("Tokenizer extended and saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--added_tokens", type=int, default=2000)
    parser.add_argument("--max_lines", type=int, default=None)

    args = parser.parse_args()

    extend_tokenizer(
        base_tokenizer_path=args.base_tokenizer,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        added_tokens=args.added_tokens,
        max_lines=args.max_lines
    )