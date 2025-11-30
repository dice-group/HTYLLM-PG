import argparse, json

from collections import Counter
from pathlib import Path
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def collect_counts(rank_dir: Path, tokenizer) -> tuple[Counter, Counter]:
    ds: Dataset = load_from_disk(str(rank_dir))
    words, subwords = Counter(), Counter()
    for txt in tqdm(ds["text"], desc=f"{rank_dir.name}", leave=False):
        sentence_words = txt.split()
        words.update(sentence_words)
        for w in sentence_words:
            subwords.update(tokenizer.tokenize(w))
    return words, subwords


def main() -> None:
    p = argparse.ArgumentParser(description="Word/subword counts for one rank folder.")
    p.add_argument("--rank-dir", type=Path, required=True)
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--out-dir", type=Path, default=Path("./rank_counts"))
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    word_counts, subword_counts = collect_counts(args.rank_dir, tok)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "word_counts.json").write_text(json.dumps(word_counts, indent=2))
    (args.out_dir / "subword_counts.json").write_text(json.dumps(subword_counts, indent=2))
    print(f"DONE {args.rank_dir.name} counts → {args.out_dir}")


if __name__ == "__main__":
    main()
