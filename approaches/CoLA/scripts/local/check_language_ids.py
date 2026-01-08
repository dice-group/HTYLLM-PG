#!/usr/bin/env python3
import argparse
from collections import Counter
import random
from pathlib import Path

from datasets import load_from_disk
from llamafactory.extras.language import load_language_groupings


def main():
    repo_root = Path(__file__).resolve().parents[2]
    tokenized_path = "/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_tiers_tokenized/llama-3.1-8B_tokenizer/cola_tier3"
    language_map_path = repo_root / "tools/two_stage_clustering/200_tier_language_groupings.json"
    split = "train"
    samples = 8
    random_samples = 256
    random_seed = 42
    full_scan = True
    strict_check = True
    strict_num_proc = 8

    ds = load_from_disk(tokenized_path)
    if isinstance(ds, dict):
        if split not in ds:
            raise SystemExit(f"split '{split}' not found in dataset")
        ds = ds[split]

    cols = ds.column_names
    print(f"columns: {cols}")
    has_lang = "language_ids" in cols
    print(f"has language_ids: {has_lang}")
    if not has_lang:
        return

    subset = ds.select(range(min(samples, len(ds))))
    for idx in range(len(subset)):
        lang_ids = subset[idx].get("language_ids")
        if lang_ids is None:
            print(f"sample {idx}: language_ids=None")
            continue
        if isinstance(lang_ids, list):
            if len(lang_ids) > 12:
                preview = lang_ids[:12]
            else:
                preview = lang_ids
        else:
            preview = lang_ids
        print(f"sample {idx}: language_ids={preview}")

    if not has_lang:
        return

    random.seed(random_seed)
    total = len(ds)
    take = min(random_samples, total)
    indices = random.sample(range(total), take)
    rnd = ds.select(indices)
    counts = Counter()
    for row in rnd:
        lang_id = row.get("language_ids")
        counts[int(lang_id)] += 1
    unique = len(counts)
    top = counts.most_common(10)
    print(f"random sample: {take} rows, unique language_ids={unique}")
    print(f"top10 language_ids: {top}")

    if not full_scan:
        return

    language_map, _, _, _ = load_language_groupings(str(language_map_path))
    if not language_map:
        raise SystemExit(f"language map not found or empty: {language_map_path}")
    languages = sorted(language_map.keys())
    language_vocab = {lang: idx for idx, lang in enumerate(languages)}

    all_sorted = None
    try:
        import pyarrow.compute as pc

        col = ds.data.column("language_ids")
        unique = pc.unique(col).to_pylist()
        all_sorted = sorted(int(v) for v in unique if v is not None)
    except Exception:
        try:
            unique = ds.unique("language_ids")
            all_sorted = sorted(int(v) for v in unique if v is not None)
        except Exception:
            all_ids = set()
            for row in ds:
                lang_id = row.get("language_ids")
                if lang_id is None:
                    continue
                all_ids.add(int(lang_id))
            all_sorted = sorted(all_ids)

    print(f"full scan: rows={len(ds)} unique_language_ids={len(all_sorted)}")
    print(f"full scan: min_id={all_sorted[0]} max_id={all_sorted[-1]}")
    preview = all_sorted[:50]
    print(f"full scan: first_50_ids={preview}")

    sample_langs = 50
    ids_to_check = all_sorted[:sample_langs]
    print(f"id->language for first_{sample_langs}: {[languages[i] for i in ids_to_check]}")

    by_id = {}
    for row in ds:
        lang_id = row.get("language_ids")
        if lang_id is None:
            continue
        lang_id = int(lang_id)
        if lang_id in by_id:
            continue
        by_id[lang_id] = row.get("language")
        if len(by_id) >= sample_langs:
            break
    print("sampled raw language strings by id:")
    for lang_id in ids_to_check:
        raw_lang = by_id.get(lang_id)
        print(f"  id {lang_id}: map={languages[lang_id]} sample={raw_lang}")

    if not strict_check:
        return

    def check_batch(batch):
        langs = batch.get("language")
        lang_ids = batch.get("language_ids")
        mismatches = []
        for lang, lang_id in zip(langs, lang_ids):
            expected = language_vocab.get(str(lang), None)
            mismatches.append(expected is None or int(lang_id) != expected)
        return {"_mismatch": mismatches}

    checked_ds = ds.map(
        check_batch,
        batched=True,
        num_proc=strict_num_proc,
        desc="strict language_id check",
        load_from_cache_file=False,
    )
    mismatches = sum(1 for v in checked_ds["_mismatch"] if v)
    print(f"strict check: checked={len(checked_ds)} mismatches={mismatches}")
    if mismatches:
        mismatch_rows = checked_ds.filter(lambda x: x["_mismatch"], num_proc=strict_num_proc)
        for i in range(min(10, len(mismatch_rows))):
            row = mismatch_rows[i]
            lang = row.get("language")
            lang_id = row.get("language_ids")
            expected = language_vocab.get(str(lang), None)
            print(f"mismatch: language={lang} language_ids={lang_id} expected={expected}")


if __name__ == "__main__":
    main()
