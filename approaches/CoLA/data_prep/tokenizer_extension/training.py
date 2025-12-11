from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm
from transformers import AutoTokenizer

from .shard_utils import find_language_shards


@dataclass
class TrainingConfig:
    data_dir: Path
    output_dir: Path
    vocab_size: int = 256_000
    base_model: str = "meta-llama/Llama-3.2-1B"
    text_key: str = "text"
    max_samples: Optional[int] = None


@dataclass
class TrainingResult:
    output_dir: Path
    vocab_size: int
    num_samples: int


def train_tokenizer(config: TrainingConfig) -> TrainingResult:
    if not config.data_dir.is_dir():
        raise ValueError(f"Data directory not found: {config.data_dir}")

    shard_paths = [path for _, path in find_language_shards(config.data_dir)]
    if not shard_paths:
        raise ValueError(f"No shard files found under {config.data_dir}")

    iterator = _TextIterator(shard_paths, config.text_key, config.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    print(f"[training] training multilingual tokenizer based on {config.base_model} architecture "
          f"on data from {config.data_dir} "
          f"using max vocab size of {config.vocab_size}"
          f"and max sample {config.max_samples}, saving to {config.output_dir}")
    new_tokenizer = tokenizer.train_new_from_iterator(tqdm(iterator), vocab_size=config.vocab_size)

    if iterator.count == 0:
        raise ValueError("No texts found for tokenizer training.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    new_tokenizer.save_pretrained(config.output_dir)

    num_samples = iterator.count
    print(
        f"[tokenize_extension] Trained tokenizer ({config.vocab_size}) saved to {config.output_dir}"
    )

    return TrainingResult(
        output_dir=config.output_dir,
        vocab_size=config.vocab_size,
        num_samples=num_samples,
    )


class _TextIterator:
    def __init__(self, shard_paths, text_key: str, max_samples: Optional[int]):
        self.shard_paths = shard_paths
        self.text_key = text_key
        self.max_samples = max_samples
        self.count = 0

    def __iter__(self) -> Iterator[str]:
        import gzip, json
        for path in self.shard_paths:
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get(self.text_key)
                    if text:
                        yield text
                        self.count += 1
                        if self.max_samples is not None and self.count >= self.max_samples:
                            return
