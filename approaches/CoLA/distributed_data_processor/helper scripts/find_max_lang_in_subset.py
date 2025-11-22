#!/usr/bin/env python
import gzip, pathlib, sys
from language_subsets import hundred_ninty_nine_representatives_mediods   # change as needed
from collections import Counter

SHARD_ROOT = pathlib.Path(
    "/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples"
)

def count_lines(file_path: pathlib.Path) -> int:
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

def max_lang_size(lang_list):
    counts = Counter()
    for lang in lang_list:
        lang_dir = SHARD_ROOT / lang
        if not lang_dir.is_dir():
            continue
        for p in lang_dir.rglob("*"):
            if p.suffix in (".jsonl", ".gz"):
                counts[lang] += count_lines(p)
    most_lang, most_cnt = counts.most_common(1)[0]
    return most_lang, most_cnt

if __name__ == "__main__":
    lang, size = max_lang_size(hundred_ninty_nine_representatives_mediods)
    print(f"Maximum language in the subset: {lang} → {size:,} sentences")
    # you can now pass `size` as the English target