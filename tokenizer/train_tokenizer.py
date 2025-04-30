# tokenizer/train_tokenizer.py
"""Train a SentencePiece‑BPE tokenizer compatible with Mixtral.

Usage
-----
$ python tokenizer/train_tokenizer.py \
        --files_glob "data/processed/*.txt" \
        --vocab_size 32000 \
        --output_dir tokenizer/
"""

import argparse
from pathlib import Path
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast


def train_tokenizer(
    files_glob: str,
    vocab_size: int = 32_000,
    output_dir: str | Path = "tokenizer",
    min_frequency: int = 2,
    special_tokens: tuple[str, str, str, str] = ("<s>", "</s>", "<unk>", "<pad>"),
) -> None:
    paths = sorted([str(p) for p in Path().glob(files_glob)])
    if not paths:
        raise ValueError(f"No files match pattern {files_glob}")

    tokenizer = SentencePieceBPETokenizer(add_prefix_space=True, dropout=0.1)
    tokenizer.train(
        files=paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        byte_fallback=True,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(output_dir / "tokenizer.json"),
        bos_token=special_tokens[0],
        eos_token=special_tokens[1],
        unk_token=special_tokens[2],
        pad_token=special_tokens[3],
    )
    hf_tok.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files_glob", required=True)
    ap.add_argument("--vocab_size", type=int, default=32_000)
    ap.add_argument("--output_dir", default="tokenizer")
    ap.add_argument("--min_frequency", type=int, default=2)
    args = ap.parse_args()
    train_tokenizer(**vars(args))