#!/usr/bin/env python
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer

# ------------------------------------------------------------------
# Existing helpers (collect_counts, build_cooccurrence_matrix, …)
# ------------------------------------------------------------------
def collect_counts(rank_dir: Path, tokenizer) -> tuple[Counter, Counter]:
    """Count words and sub‑words **inside a single rank directory**."""
    ds: Dataset = load_from_disk(str(rank_dir))          # ← now a valid HF dataset
    word_counts, subword_counts = Counter(), Counter()

    for txt in tqdm(ds["text"], desc=f"Counting in {rank_dir.name}", leave=False):
        words = txt.split()
        word_counts.update(words)
        for w in words:
            sub_tokens = tokenizer.tokenize(w)
            subword_counts.update(sub_tokens)
    return word_counts, subword_counts

# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute word / sub‑word frequencies for ONE rank folder."
    )
    parser.add_argument("--rank-dir", type=Path, required=True,
                        help="Path to a single rank_<N> directory.")
    parser.add_argument("--tokenizer", type=str,
                        default="meta-llama/Llama-3.1-8B",
                        help="HF tokenizer name.")
    parser.add_argument("--out-dir", type=Path, default=Path("./rank_counts"),
                        help="Where to write word_counts.json / subword_counts.json.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # ---- counting ----------------------------------------------------
    word_counts, subword_counts = collect_counts(args.rank_dir, tokenizer)

    # ---- write results ------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "word_counts.json").write_text(json.dumps(word_counts, indent=2))
    (args.out_dir / "subword_counts.json").write_text(json.dumps(subword_counts, indent=2))
    print(f"✅ Finished {args.rank_dir.name} → {args.out_dir}")

if __name__ == "__main__":
    main()