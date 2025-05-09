#!/usr/bin/env python
"""
Tokenise *once* and write a ready‑to‑train Arrow dataset to disk.
"""

from __future__ import annotations
import argparse, logging, gzip, json
from pathlib import Path
from typing import List

from datasets import load_dataset, Features, Value, Dataset
from transformers import AutoTokenizer


def chunk_batch(texts: List[str], tokenizer, seq_len: int):
    """Return {'input_ids': [[...],[...]]}; keeps only *full‑length* chunks."""
    all_ids = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        usable = (len(ids) // seq_len) * seq_len
        for i in range(0, usable, seq_len):
            all_ids.append(ids[i : i + seq_len])
    return {"input_ids": all_ids}



def get_raw_dataset(pattern: str, text_field: str):
    files = sorted(str(p) for p in Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")

    # generator that yields *only* the text column
    def line_gen():
        for fp in files:
            opener = gzip.open if fp.endswith(".gz") else open
            with opener(fp, "rt", encoding="utf-8") as f:
                for ln in f:
                    obj = json.loads(ln)
                    txt = obj.get(text_field, "")
                    if txt:                      # skip blanks
                        yield {"text": txt}

    features = Features({"text": Value("string")})
    return Dataset.from_generator(line_gen, features=features)



def main():
    ap = argparse.ArgumentParser("pre‑tokenise corpus")
    ap.add_argument("--files", required=True,
                    help='glob, e.g. "data/corpus/*.jsonl.gz"')
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--seq_length", type=int, default=1024)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--text_field", default="text",
                    help="name of the JSON field that contains the document text")
    ap.add_argument("--num_proc", type=int, default=None,
                    help="tokenisation workers (defaults to cpu_count())")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    raw = get_raw_dataset(args.files, args.text_field)

    # ── 1️⃣ throw away docs that are *too short* for even 1 full block
    raw = raw.filter(
        lambda ex: len(tok(ex["text"], add_special_tokens=False).input_ids)+1
                   >= args.seq_length,
        num_proc=args.num_proc,
    )

    # ── 2️⃣ cut every doc into fixed‑length sequences
    tokenised = raw.map(
        lambda batch: chunk_batch(batch["text"], tok, args.seq_length),
        batched=True,
        remove_columns=["text"],
        num_proc=args.num_proc,
    )

    # Arrow supports memory‑mapping, so we can save massive datasets safely
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tokenised.save_to_disk(args.out_dir)
    logging.info(f"✅ wrote {len(tokenised):,} sequences to {args.out_dir}")


if __name__ == "__main__":
    main()
