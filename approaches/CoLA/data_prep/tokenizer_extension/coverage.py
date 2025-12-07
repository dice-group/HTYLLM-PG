from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class CoverageConfig:
    data_dir: Path
    tokenizer: str
    max_lines: int = 10_000
    num_workers: Optional[int] = None
    output_csv: Optional[Path] = None


def compute_coverage(config: CoverageConfig) -> pd.DataFrame:
    files = sorted(_collect_files(config.data_dir))

    tasks = [
        (path, config.tokenizer, config.max_lines)
        for path in files
    ]
    workers = config.num_workers or cpu_count()

    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_compute_file_metrics, tasks),
                total=len(tasks),
                desc="coverage",
            )
        )

    df = pd.DataFrame(results).sort_values("language").reset_index(drop=True)
    if config.output_csv:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.output_csv, index=False)
    return df


def _collect_files(data_dir: Path) -> Iterable[Path]:
    for entry in data_dir.iterdir():
        if entry.is_file() and entry.name.endswith(".jsonl.gz"):
            yield entry


def _read_jsonl_gz(path: Path, max_lines: int) -> List[str]:
    texts: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if max_lines and i >= max_lines:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = payload.get("text") or payload.get("content") or ""
            if text:
                texts.append(text)
    return texts


def _compute_file_metrics(args: tuple[Path, str, int]) -> dict:
    path, tokenizer_name, max_lines = args
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    texts = _read_jsonl_gz(path, max_lines=max_lines)

    total_chars = total_words = total_tokens = unk_count = 0
    for text in texts:
        total_chars += len(text)
        words = text.split()                #TODO: find appropriate method for multilingual data
        total_words += len(words)
        tokens = tokenizer(text)["input_ids"]
        total_tokens += len(tokens)
        if tokenizer.unk_token_id is not None:
            unk_count += tokens.count(tokenizer.unk_token_id)

    chars_per_token = total_chars / total_tokens if total_tokens else 0.0
    pieces_per_word = total_tokens / total_words if total_words else 0.0
    unknown_rate = unk_count / total_tokens if total_tokens else 0.0

    return {
        "language": path.stem.replace(".jsonl", ""),
        "chars_per_token": chars_per_token,
        "pieces_per_word": pieces_per_word,
        "unknown_rate": unknown_rate,
        "num_texts": len(texts),
    }

