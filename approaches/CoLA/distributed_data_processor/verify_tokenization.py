import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_from_disk


def _load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _list_datasets(dataset_dir: Path) -> List[Path]:
    rank_dirs = sorted(
        path for path in dataset_dir.iterdir() if path.is_dir() and path.name.startswith("rank_")
    )
    return rank_dirs or [dataset_dir]


def verify_counts(manifest_path: Path, dataset_dir: Path) -> None:
    manifest = _load_manifest(manifest_path)
    expected = manifest.get("total_samples")
    if expected is None:
        raise RuntimeError("'total_samples' missing from manifest.")

    dataset_paths = _list_datasets(dataset_dir)
    total = 0
    for path in dataset_paths:
        ds = load_from_disk(str(path))
        count = len(ds)
        total += count
        print(f"{path.name}: {count} samples")

    print(f"Total samples found: {total}")
    print(f"Expected samples from manifest: {expected}")
    if total != expected:
        raise SystemExit(f"Mismatch detected (found={total}, expected={expected}).")
    print("Verification successful: counts match.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to shard_manifest.json produced during sharding.")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing rank_*/ tokenized shards or a combined dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    dataset_dir = Path(args.dataset_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} does not exist.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    verify_counts(manifest_path, dataset_dir)


if __name__ == "__main__":
    main()
