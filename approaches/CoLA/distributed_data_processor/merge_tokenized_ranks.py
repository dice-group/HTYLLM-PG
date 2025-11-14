import argparse
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def merge_pair(args):
    a, b, out = args
    da = load_from_disk(str(a))
    db = load_from_disk(str(b))
    merged = concatenate_datasets([da, db])
    merged.save_to_disk(str(out))
    return out


def discover_rank_dirs(root: Path) -> list[Path]:
    ranks = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("rank_"))
    if not ranks:
        raise RuntimeError("No rank_* dirs found")
    return ranks


def _count_stages(total: int) -> int:
    count = 0
    current = total
    while current > 1:
        count += 1
        current = (current // 2) + (current % 2)
    return max(count, 0)


def pairwise_merge_concurrent(dirs: list[Path], tmp_root: Path, max_workers: int = 4) -> Path:
    current = dirs
    stage = 0
    total_stages = _count_stages(len(current))

    total_initial = len(current)
    while len(current) > 1:
        stage += 1
        tasks = []
        next_round = []
        stage_dir = tmp_root / f"stage_{stage:03d}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(current), 2):
            if i + 1 == len(current):
                next_round.append(current[i])
                break

            a, b = current[i], current[i + 1]
            pair_idx = i // 2
            out = stage_dir / f"pair_{pair_idx:05d}"
            tasks.append((a, b, out))

        remaining_pct = (len(current) / total_initial) * 100
        if tasks:
            print(f"[merge][stage {stage}/{total_stages}] start: {len(tasks)} pairs, {len(current)} datasets ({remaining_pct:.1f}%).")
        completed = 0
        total_pairs = len(tasks)
        if tasks:
            if max_workers <= 1:
                for task in tasks:
                    out_dir = merge_pair(task)
                    completed += 1
                    pct_pairs = (completed / total_pairs) * 100 if total_pairs else 100
                    print(f"[merge][stage {stage}/{total_stages}] pair {completed}/{total_pairs} done ({pct_pairs:.1f}%).")
                    next_round.append(out_dir)
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    for out_dir in ex.map(merge_pair, tasks):
                        completed += 1
                        pct_pairs = (completed / total_pairs) * 100 if total_pairs else 100
                        print(f"[merge][stage {stage}/{total_stages}] pair {completed}/{total_pairs} done ({pct_pairs:.1f}%).")
                        next_round.append(out_dir)

        current = next_round
        remaining_pct = (len(current) / total_initial) * 100 if total_initial else 0
        print(f"[merge][stage {stage}/{total_stages}] end: {len(current)} datasets remain ({remaining_pct:.1f}%).")

    return current[0]


def _build_dataset_dict(dataset, val_fraction: float, seed: int) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset

    if "split" in dataset.column_names:
        print("[split] Using existing split column to build validation set.")
        validation_ds = dataset.filter(lambda ex: ex["split"] == "validation")
        train_ds = dataset.filter(lambda ex: ex["split"] != "validation")
        return DatasetDict(train=train_ds, validation=validation_ds)

    print(f"[split] No split column found; falling back to random split fraction={val_fraction:.3f}.")
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def merge_ranks(root: Path, output: Path, overwrite: bool, workers: int, split_fraction: float, split_seed: int):
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"{output} exists. Use --overwrite.")
        shutil.rmtree(output)

    ranks = discover_rank_dirs(root)

    with tempfile.TemporaryDirectory() as tmp:
        final_dir = pairwise_merge_concurrent(ranks, Path(tmp), workers)
        if split_fraction > 0:
            merged_dataset = load_from_disk(str(final_dir))
            dataset_dict = _build_dataset_dict(merged_dataset, split_fraction, split_seed)
            for split_name, split_dataset in dataset_dict.items():
                cols = [col for col in ["language", "split"] if col in split_dataset.column_names]
                if cols:
                    dataset_dict[split_name] = split_dataset.remove_columns(cols)
            dataset_dict.save_to_disk(str(output))
            print(f"Merged dataset with validation split saved to {output}")
        else:
            shutil.copytree(final_dir, output)
            print(f"Merged dataset saved to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized_root", type=Path, required=True)
    ap.add_argument("--output_path", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split_fraction", type=float, default=0.05, help="Fraction to reserve for validation (set 0 to skip split).")
    ap.add_argument("--split_seed", type=int, default=42)
    args = ap.parse_args()

    merge_ranks(
        args.tokenized_root,
        args.output_path,
        args.overwrite,
        args.workers,
        args.split_fraction,
        args.split_seed,
    )


if __name__ == "__main__":
    main()
