#!/usr/bin/env python
"""
Scan the tokenized output tree, load each Hugging Face dataset, and dump basic stats.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_from_disk

OUTPUT_BASE = Path("/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter")
REPORT_PATH = Path("logs/dataset_report.md")
MAX_TOKEN_SAMPLES = 0  # 0 = use whole dataset, otherwise sample this many rows to estimate tokens
VERBOSE = True


def find_datasets(base_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    for tokenizer_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for subset_dir in sorted(p for p in tokenizer_dir.iterdir() if p.is_dir()):
            if (subset_dir / "dataset_info.json").exists():
                yield tokenizer_dir.name, subset_dir.name, subset_dir


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total += (Path(root) / fname).stat().st_size
    return total


def summarize_dataset(
    ds_path: Path,
    max_token_samples: int,
    verbose: bool = False,
) -> Dict[str, float]:
    ds = load_from_disk(str(ds_path))
    num_rows = len(ds)

    # Token counting (optionally approximate to speed things up)
    token_sum = 0
    sampled = 0
    target_samples = max_token_samples if max_token_samples > 0 else num_rows
    for sample in ds:
        input_ids = sample.get("input_ids") or sample.get("input_ids".encode())  # safety
        if isinstance(input_ids, bytes):
            input_ids = json.loads(input_ids)
        token_sum += len(input_ids) if input_ids is not None else 0
        sampled += 1
        if sampled >= target_samples:
            break

    if sampled and sampled < num_rows:
        scale = num_rows / sampled
        token_sum = int(token_sum * scale)

    size_bytes = dir_size_bytes(ds_path)
    avg_tokens = (token_sum / num_rows) if num_rows else 0

    if verbose:
        print(
            f"[verify] {ds_path}: rows={num_rows}, tokens≈{token_sum}, "
            f"avg_tokens={avg_tokens:.1f}, size_gb={size_bytes / 1e9:.2f}"
        )

    return {
        "rows": num_rows,
        "tokens": token_sum,
        "avg_tokens": avg_tokens,
        "size_gb": size_bytes / 1e9,
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def render_markdown(stats: Dict[str, List[Tuple[str, Path, Dict[str, float]]]]) -> str:
    lines: List[str] = []
    lines.append("# Tokenized Dataset Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat()}_")
    lines.append("")

    for tokenizer in sorted(stats):
        lines.append(f"## {tokenizer}")
        lines.append("")
        for subset, path, metrics in sorted(stats[tokenizer], key=lambda x: x[0]):
            lines.append(f"<details><summary>{subset} — {metrics['rows']} samples</summary>")
            lines.append("")
            lines.append(f"- Path: `{path}`")
            lines.append(f"- Samples: {metrics['rows']:,}")
            lines.append(f"- Tokens (approx): {metrics['tokens']:,}")
            lines.append(f"- Avg tokens/sample: {metrics['avg_tokens']:.1f}")
            lines.append(f"- Disk size: {metrics['size_gb']:.2f} GB")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    stats: Dict[str, List[Tuple[str, Path, Dict[str, float]]]] = defaultdict(list)

    for tokenizer_name, subset_name, ds_path in find_datasets(OUTPUT_BASE):
        metrics = summarize_dataset(ds_path, MAX_TOKEN_SAMPLES, verbose=VERBOSE)
        stats[tokenizer_name].append((subset_name, ds_path, metrics))

    report_md = render_markdown(stats)
    ensure_parent(REPORT_PATH)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"[verify] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
