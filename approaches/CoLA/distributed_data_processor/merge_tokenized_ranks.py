import argparse
import shutil
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def _discover_rank_dirs(root: Path) -> list[Path]:
    rank_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("rank_"))
    if not rank_dirs:
        dataset_marker = root / "dataset_info.json"
        if dataset_marker.exists():
            return [root]
        raise RuntimeError(f"No rank_* directories or dataset found under {root}. Nothing to merge.")
    return rank_dirs


def merge_ranks(input_root: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise RuntimeError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    rank_dirs = _discover_rank_dirs(input_root)
    datasets = []
    for rank_dir in rank_dirs:
        print(f"Loading dataset from {rank_dir}")
        datasets.append(load_from_disk(str(rank_dir)))

    print(f"Concatenating {len(datasets)} rank datasets")
    combined = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(output_path))
    print(f"Combined dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge tokenized rank_* datasets into one Dataset.")
    parser.add_argument("--tokenized_root", type=Path, required=True, help="Directory containing rank_* subdirectories.")
    parser.add_argument("--output_path", type=Path, required=True, help="Directory to save the merged dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Allow existing output_path to be overwritten.")
    args = parser.parse_args()

    merge_ranks(args.tokenized_root, args.output_path, args.overwrite)


if __name__ == "__main__":
    main()
