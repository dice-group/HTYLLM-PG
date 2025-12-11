import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List
from datasets import Dataset, DatasetDict, load_from_disk


def _eligible_languages(ds: Dataset, lang_column: str, min_size: int) -> Dict[str, int]:
    counts = Counter(ds[lang_column])
    return {lang: cnt for lang, cnt in counts.items() if cnt >= min_size}


def _top_indices_for_language(lang_array: np.ndarray, score_array: np.ndarray, language: str, limit: int) -> np.ndarray:
    lang_idx = np.where(lang_array == language)[0]
    if len(lang_idx) == 0:
        return np.empty(0, dtype=np.int64)
    if len(lang_idx) <= limit:
        return np.sort(lang_idx)
    local_scores = score_array[lang_idx]
    selected = np.argpartition(-local_scores, limit - 1)[:limit]
    return np.sort(lang_idx[selected])


def _filter_split(ds: Dataset, eligible_langs: Iterable[str], limit: int, lang_column: str, score_column: str,) -> Dataset:
    lang_arr = np.array(ds[lang_column])
    score_arr = np.array(ds[score_column], dtype=np.float32)
    keep_indices: List[int] = []
    for lang in eligible_langs:
        selected = _top_indices_for_language(lang_arr, score_arr, lang, limit)
        keep_indices.extend(selected.tolist())
    if not keep_indices:
        return ds.select([])
    keep_indices.sort()
    return ds.select(keep_indices)


def _filter_validation(ds: Dataset, eligible_langs: Iterable[str], lang_column: str) -> Dataset:
    lang_arr = np.array(ds[lang_column])
    mask = np.isin(lang_arr, np.array(list(eligible_langs)))
    idx = np.where(mask)[0]
    if not len(idx):
        return ds.select([])
    return ds.select(idx.tolist())


def _format_suffix(size: int) -> str:
    if size % 1000 == 0:
        return f"top{size // 1000}k"
    return f"top{size}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create top-K-per-language merged datasets.")
    parser.add_argument("--merged-root", type=Path, required=True, help="Path to the merged DatasetDict.")
    parser.add_argument("--output-prefix", type=str, required=True, help="Base path prefix for the filtered datasets.")
    parser.add_argument("--sizes", type=int, nargs="+", required=True, help="List of top-K sizes (e.g., 10000 20000 30000).")
    parser.add_argument("--min-language-size", type=int, default=30000, help="Minimum number of train examples required per language.")
    parser.add_argument("--train-split", type=str, default="train", help="Name of the train split inside the dataset.")
    parser.add_argument("--lang-column", type=str, default="language", help="Column that stores the language id.")
    parser.add_argument("--score-column", type=str, default="joint_score", help="Column containing the ranking score.")
    args = parser.parse_args()
    sizes = [s for s in args.sizes if s > 0]
    if not sizes:
        raise RuntimeError("Provide at least one positive integer via --sizes.")

    data = load_from_disk(str(args.merged_root))
    if isinstance(data, DatasetDict):
        ds_dict = data
    else:
        if "split" in data.column_names:
            ds_dict = DatasetDict(
                train=data.filter(lambda x: x.get("split") != "validation"),
                validation=data.filter(lambda x: x.get("split") == "validation"),
            )
        else:
            ds_dict = DatasetDict(train=data)
    if args.train_split not in ds_dict:
        raise RuntimeError(f"Split '{args.train_split}' was not found in {args.merged_root}.")
    train_ds = ds_dict[args.train_split]

    if args.lang_column not in train_ds.column_names:
        raise RuntimeError(f"Column '{args.lang_column}' missing from train split.")
    if args.score_column not in train_ds.column_names:
        raise RuntimeError(
            f"Column '{args.score_column}' missing. Did you run the ranking step before merging?"
        )

    eligible = _eligible_languages(train_ds, args.lang_column, args.min_language_size)
    if not eligible:
        raise RuntimeError(
            f"No languages have at least {args.min_language_size} train examples. "
            "Lower --min-language-size or ensure ranking ran on the expected corpus."
        )
    print(f"[INFO] Eligible languages ({len(eligible)}): {', '.join(sorted(eligible))}")

    for size in sorted(set(sizes)):
        langs_for_size = [lang for lang, cnt in eligible.items() if cnt >= size]
        if not langs_for_size:
            print(f"[WARN] No language has >= {size} examples; skipping top-{size}.")
            continue

        print(f"[INFO] Building top-{size} dataset for {len(langs_for_size)} languages.")
        new_splits = {}
        new_splits[args.train_split] = _filter_split(
            train_ds,
            langs_for_size,
            size,
            args.lang_column,
            args.score_column,
        )
        for split_name, split_ds in ds_dict.items():
            if split_name == args.train_split:
                continue
            if args.lang_column not in split_ds.column_names:
                new_splits[split_name] = split_ds
                continue
            new_splits[split_name] = _filter_validation(split_ds, langs_for_size, args.lang_column)

        out_path = Path(f"{args.output_prefix}_{_format_suffix(size)}")
        DatasetDict(new_splits).save_to_disk(str(out_path))
        count_str = new_splits[args.train_split].num_rows
        print(f"[INFO] Saved {count_str:,} rows to {out_path}")


if __name__ == "__main__":
    main()
