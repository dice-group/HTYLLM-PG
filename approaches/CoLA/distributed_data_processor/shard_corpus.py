import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Iterator


def iter_source_files(source_dir: Path) -> Iterator[Path]:
    for root, _, files in os.walk(source_dir):
        for name in sorted(files):
            if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
                yield Path(root) / name


def iter_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    kwargs = {"encoding": "utf-8", "errors": "ignore"}
    with opener(path, mode, **kwargs) as handle:
        for line in handle:
            yield line.rstrip("\n")


def shard_corpus(source_dir: Path, shard_dir: Path, target_bytes: int) -> dict:
    shard_dir.mkdir(parents=True, exist_ok=True)
    if any(shard_dir.iterdir()):
        raise RuntimeError(f"Output directory {shard_dir} is not empty. Provide an empty path.")

    manifest = {
        "source_dir": str(source_dir.resolve()),
        "shard_dir": str(shard_dir.resolve()),
        "target_shard_bytes": target_bytes,
        "total_samples": 0,
        "total_shards": 0,
        "shards": [],
    }

    shard_idx = 0
    writer = None
    shard_count = 0
    shard_bytes = 0

    def open_writer(index: int):
        shard_path = shard_dir / f"shard_{index:06d}.jsonl.gz"
        return gzip.open(shard_path, "wt", encoding="utf-8"), shard_path

    shard_path = None

    try:
        for src_path in iter_source_files(source_dir):
            for line in iter_lines(src_path):
                encoded = line.encode("utf-8")
                line_bytes = len(encoded) + 1  # newline
                if writer is None or shard_bytes + line_bytes > target_bytes:
                    if writer is not None:
                        writer.close()
                        manifest["shards"].append(
                            {
                                "path": shard_path.name,
                                "samples": shard_count,
                                "bytes": shard_bytes,
                            }
                        )
                        manifest["total_shards"] += 1
                    writer, shard_path = open_writer(shard_idx)
                    shard_idx += 1
                    shard_count = 0
                    shard_bytes = 0

                writer.write(line)
                writer.write("\n")
                shard_count += 1
                shard_bytes += line_bytes
                manifest["total_samples"] += 1

        if writer is not None:
            writer.close()
            manifest["shards"].append(
                {"path": shard_path.name, "samples": shard_count, "bytes": shard_bytes}
            )
            manifest["total_shards"] += 1

    finally:
        if writer is not None and not writer.closed:
            writer.close()

    if manifest["total_samples"] == 0:
        raise RuntimeError(f"No samples were found in {source_dir}.")

    manifest_path = shard_dir / "shard_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"Wrote {manifest['total_shards']} shard(s) with {manifest['total_samples']} samples "
        f"to {shard_dir}. Manifest saved to {manifest_path}."
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing language subdirectories.")
    parser.add_argument("--shard_dir", required=True, help="Destination directory for the uniform shards.")
    parser.add_argument("--target_shard_bytes", type=int, default=512 * 1024 * 1024, help="Approximate size for each shard in bytes (default: 512MB).")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    shard_dir = Path(args.shard_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    shard_corpus(source_dir, shard_dir, args.target_shard_bytes)


if __name__ == "__main__":
    main()
