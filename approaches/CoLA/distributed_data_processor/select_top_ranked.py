#!/usr/bin/env python
"""
select_top_ranked.py
-------------------
Select the top-Q% (or a fixed number) of sentences from the
merged tokenized dataset that have the highest *joint* rank score
(Rj) computed by Algorithm 2 in the paper.

The script writes a JSON-Lines file that will be fed to the CPT
trainer.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict


def _load_rank_scores(ds: Dataset) -> np.ndarray:
    """
    The merge script already stores a column called ``joint_score`` that
    contains the joint score (Rj) for each sentence.  If you used a
    different column name, change the key below.
    """
    if "joint_score" not in ds.column_names:
        raise RuntimeError(
            "Merged dataset does not contain a `joint_score` column. "
            "Make sure you ran the ranking step before merging."
        )
    return np.array(ds["joint_score"])


def select_top(
    merged_path: Path,
    output_path: Path,
    size: int = 20_000,
    percentile: float | None = None,
) -> None:
    """
    Parameters
    ----------
    merged_path: Path
        Directory that contains the merged tokenized DatasetDict
        (the output of ``merge_tokenized_ranks.py``).
    output_path: Path
        Destination ``.jsonl`` file that will hold the CPT corpus.
    size: int, optional
        Absolute number of examples to keep (default = 20k).
    percentile: float, optional
        If given, selects the top ``percentile``% of the data instead
        of a fixed ``size`` (mutually exclusive with ``size``).
    """
    # Load the merged dataset
    ds_dict: DatasetDict = load_from_disk(str(merged_path))
    # We usually fine‑tune on the *train* split; if none exists we fall back
    # to the whole dataset.
    if "train" in ds_dict:
        ds = ds_dict["train"]
    else:
        ds = ds_dict["dataset"] if "dataset" in ds_dict else list(ds_dict.values())[0]

    # Pull the joint rank scores (Rj) that were computed in
    # Algorithm 2 of the CPT paper.
    scores = _load_rank_scores(ds)

    # Determine which indices to keep
    if percentile is not None:
        # keep the top <percentile> % of examples
        k = int(len(scores) * percentile / 100.0)
    else:
        k = min(size, len(scores))

    top_idx = np.argpartition(-scores, k - 1)[:k]   # fastest “top‑k”

    # Preserve the original order (optional, but easier to read)
    top_idx = sorted(top_idx.tolist())

    # Write the selected rows as JSON‑Lines
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for i in top_idx:
            # ``ds[i]`` is a dict; keep only the fields we need for CPT
            row = {
                "input_ids": ds[i]["input_ids"],
                "attention_mask": ds[i]["attention_mask"],
                "labels": ds[i]["input_ids"],
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Selected {k:,} examples → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick the top‑ranked sentences for CPT.")
    parser.add_argument(
        "--merged_root",
        type=Path,
        required=True,
        help="Path to the merged tokenized dataset (output of merge_tokenized_ranks.py).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="File that will contain the CPT corpus (JSON‑Lines).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--size",
        type=int,
        default=20_000,
        help="Number of examples to keep (default = 20k).",
    )
    group.add_argument(
        "--percentile",
        type=float,
        help="Select the top X%% of examples instead of a fixed size.",
    )
    args = parser.parse_args()
    select_top(args.merged_root, args.output_path, size=args.size, percentile=args.percentile)


if __name__ == "__main__":
    main()