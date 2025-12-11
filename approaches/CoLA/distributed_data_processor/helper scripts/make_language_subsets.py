#!/usr/bin/env python
"""
make_language_subsets.py
------------------------
Wrapper that builds a balanced English shard.
"""

import gzip
import shutil
from pathlib import Path
from typing import List, Dict
from collections import Counter
import sys

# Helper: count lines (i.e. sentences) in a shard file
def count_lines(p: Path) -> int:
    """Return the number of newline‑separated sentences in p."""
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

# Sample English shards until a target sentence count is reached
def sample_english_shards(
    english_root: Path,
    target_sentences: int,
) -> List[Path]:
    """
    Walk the English shard folder (sorted for reproducibility) and keep
    files until the cumulative line count >= target_sentences.
    Returns the list of selected shard Paths.
    """
    selected: List[Path] = []
    cum = 0
    for shard in sorted(p for p in english_root.rglob("*")
                       if p.is_file() and p.suffix in (".jsonl", ".gz")):
        if cum >= target_sentences and selected:
            break
        selected.append(shard)
        cum += count_lines(shard)
    print(f"Selected {len(selected)} English shards → {cum:,} sentences "
          f"(target {target_sentences:,})")
    return selected

def main() -> None:
    # Adapt on how many English sentences you want in the final
    # balanced English shard.
    TARGET_SENTENCES = [600000,4500000,7000000]         # <--- change this number only

    # Paths (adapt if your directory layout differs)
    SHARD_ROOT   = Path("/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples")
    ENGLISH_ROOT = SHARD_ROOT / "english"                     # English shards
    for ts in TARGET_SENTENCES:
        OUT_ROOT     = Path(f"/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples/english_{ts}_sentences")\
                        .resolve()

        # Build the English subset with the user‑specified sentence count
        selected_shards = sample_english_shards(ENGLISH_ROOT, ts)

        # Write the selected shards to a fresh folder.
        # Downstream you can point tokenize_slurm.py to this folder via
        # --languages or add it to a *_plus_english* list.
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        for src in selected_shards:
            dst = OUT_ROOT / src.name
            shutil.copy2(src, dst)

        print(f"\nEnglish shard written to {OUT_ROOT}\n")

if __name__ == "__main__":
    main()