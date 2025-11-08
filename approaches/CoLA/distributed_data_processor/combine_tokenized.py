import argparse
import json
import shutil
from pathlib import Path
from typing import List

from datasets import concatenate_datasets, load_from_disk


def _list_rank_dirs(parent: Path) -> List[Path]:
    rank_dirs = sorted(
        path for path in parent.iterdir() if path.is_dir() and path.name.startswith("rank_")
    )
    return rank_dirs or [parent]


def combine(tokenized_dir: Path, output_dir: Path, manifest_path: Path | None, overwrite: bool) -> int:
    if output_dir.exists():
        if not overwrite:
            raise RuntimeError(f"Output directory {output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    rank_dirs = _list_rank_dirs(tokenized_dir)
    datasets = [load_from_disk(str(rank_dir)) for rank_dir in rank_dirs]
    combined = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    combined.save_to_disk(str(output_dir))
    total = len(combined)
    print(f"Combined dataset saved to {output_dir} with {total} samples.")

    if manifest_path is not None:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = manifest.get("total_samples")
        if expected is None:
            raise RuntimeError(f"Manifest {manifest_path} is missing 'total_samples'.")
        if total != expected:
            raise SystemExit(f"Sample mismatch: combined={total}, manifest={expected}.")
        print("Manifest verification successful: counts match.")

    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Combine rank tokenized shards into a single dataset.")
    parser.add_argument("--tokenized_dir", required=True, help="Directory containing rank_*/ shards.")
    parser.add_argument("--output_dir", required=True, help="Destination for the combined dataset.")
    parser.add_argument("--manifest", default=None, help="Path to shard_manifest.json for verification.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output_dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenized_dir = Path(args.tokenized_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else None
    combine(tokenized_dir, output_dir, manifest_path, args.overwrite)


if __name__ == "__main__":
    main()
