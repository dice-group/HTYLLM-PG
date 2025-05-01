"""Train a SentencePiece‑BPE tokenizer directly from **compressed JSON‑Lines** or plain text.

Examples
--------
# JSONL.GZ corpus (each line has a `text` field)
$ python tokenizer/train_tokenizer.py \
        --files_glob "data/fineweb/*.jsonl.gz" \
        --vocab_size 131072

# Plain‑text shards (one document per file)
$ python tokenizer/train_tokenizer.py --files_glob "data/txt/*.txt"
"""

from __future__ import annotations

import argparse, gzip, json, itertools, os
from pathlib import Path
from typing import Iterable

from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

TXT_EXTS = {".txt", ".text"}
JSON_EXTS = {".jsonl", ".jsonl.gz", ".json", ".json.gz"}


def stream_jsonl(files: list[str], field: str = "text") -> Iterable[str]:
    """Yield `field` from each JSON object in all files (supports gzip)."""
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if field in obj and obj[field].strip():
                        yield obj[field]
                except json.JSONDecodeError:
                    continue


def train_tokenizer(
    files_glob: str,
    vocab_size: int = 131_072,
    output_dir: str | Path = "tokenizer",
    min_frequency: int = 2,
    json_field: str = "text",
    special_tokens: tuple[str, str, str, str] = ("<s>", "</s>", "<unk>", "<pad>"),
) -> None:
    paths = sorted([str(p) for p in Path().glob(files_glob)])
    if not paths:
        raise ValueError(f"No files match pattern {files_glob}")
    
    # Debug information
    print(f"Found {len(paths)} paths matching pattern {files_glob}")
    for p in paths[:10]:  # Print first 10 paths
        print(f"Path: {p}, Suffix: {Path(p).suffix.lower()}")
    
    # Decide training mode
    if all(Path(p).suffix.lower() in TXT_EXTS for p in paths):
        # ---------- plain‑text path training ----------
        print("Using plain-text training mode")
        tokenizer = SentencePieceBPETokenizer(add_prefix_space=True, dropout=0.1)
        tokenizer.train(
            files=paths,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            byte_fallback=True,
        )
    elif all(Path(p).name.lower().endswith(('.jsonl.gz', '.jsonl', '.json.gz', '.json')) for p in paths):
        # ---------- JSONL / JSONL.GZ training ----------
        print("Using JSON training mode")
        tokenizer = SentencePieceBPETokenizer(add_prefix_space=True, dropout=0.1)
        iterator = stream_jsonl(paths, json_field)
        tokenizer.train_from_iterator(
            iterator,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
        )
    else:
        print("Extension check failed. Printing all paths and their extensions:")
        for p in paths:
            print(f"Path: {p}, Suffix: {Path(p).suffix.lower()}, Name: {Path(p).name}")
        raise ValueError("Mixed or unrecognised extensions in corpus.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(out / "tokenizer.json"),
        bos_token=special_tokens[0],
        eos_token=special_tokens[1],
        unk_token=special_tokens[2],
        pad_token=special_tokens[3],
    )
    hf_tok.save_pretrained(out)
    print("Tokenizer saved to", out.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files_glob", required=True)
    ap.add_argument("--vocab_size", type=int, default=131_072)
    ap.add_argument("--output_dir", default="tokenizer")
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--json_field", default="text", help="Field name containing text inside JSON lines")
    args = ap.parse_args()
    train_tokenizer(**vars(args))