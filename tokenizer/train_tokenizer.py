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
from tokenizers.trainers import SentencePieceBpeTrainer
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

def chunk(seq, size):
    """Yield successive `size`-item chunks from any sequence or list."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]




def train_tokenizer(
    files_glob: str,
    vocab_size: int = 131_072,
    output_dir: str | Path = "tokenizer",
    min_frequency: int = 2,
    json_field: str = "text",
    special_tokens: tuple[str, str, str, str] = ("<s>", "</s>", "<unk>", "<pad>"),
    chunk_size: int = 500,              # NEW parameter to accept CLI flag
) -> None:
    import glob, os
    if os.path.isabs(files_glob):
        # absolute path → use glob (supports **)
        paths = sorted(glob.glob(files_glob, recursive=True))
    else:
        # relative path → pathlib is fine (keeps old behaviour)
        paths = sorted(str(p) for p in Path().glob(files_glob))
    
    # Check file extensions to determine processing mode
    any_ext = Path(paths[0]).suffix if paths else ""
    
    if any(ext in JSON_EXTS for ext in (any_ext, any_ext + ".gz")):
        # ---------- JSONL / JSONL.GZ training (streamed) ----------
        print("Using JSON streaming mode to keep RAM low")

        # 1️⃣  Build one SentencePiece trainer you will reuse
        sp_trainer = SentencePieceBpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
            byte_fallback=True,
        )
        tokenizer = SentencePieceBPETokenizer(add_prefix_space=True, dropout=0.0)

        # 2️⃣  Loop over file shards
        for shard in chunk(paths, chunk_size):
            print(f" → Training on {len(shard)} shards "
                f"({Path(shard[0]).name} … {Path(shard[-1]).name})")

            # 3️⃣  Lazy line generator for just this shard
            def lines():
                for doc in stream_jsonl(shard, json_field):
                    yield doc

            # 4️⃣  We *must* pass `length=` so the Rust core does not
            #     buffer everything first.  A loose upper bound is fine.
            tokenizer.train_from_iterator(
                lines(),
                trainer=sp_trainer,
                length=100_000_000,        # adjust to your corpora
                show_progress=True,
            )
    
    elif any_ext.lower() in TXT_EXTS:
        # ---------- Plain text training ----------
        tokenizer = SentencePieceBPETokenizer(add_prefix_space=True)
        tokenizer.train(
            files=paths,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
        )

    # 5️⃣  Save once at the end
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
    ap.add_argument("--chunk_size", type=int, default=500,
                help="Number of files to train on at once")
    args = ap.parse_args()
    
    # Keep the Rust core single-process to save RAM
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    train_tokenizer(**vars(args))

# Notes on memory optimization:
# -----------------------------
# * `length=100_000_000` tells the Rust core roughly how many sentences to expect,
#   so it never tries to buffer the whole iterator
# * One `SentencePieceBpeTrainer` reused across shards means every call to
#   `train_from_iterator` continues training, instead of overwriting the merge table
# * Dataset streaming keeps memory usage low since data isn't downloaded at once
# * Disabling tokenizer parallelism with TOKENIZERS_PARALLELISM=false prevents
#   the Rust backend from forking workers that each keep a copy of the pair matrix
#   which can save several GB of memory