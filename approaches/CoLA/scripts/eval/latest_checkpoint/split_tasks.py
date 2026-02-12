#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def _read_tasks(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text().splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def _write_chunks(items: list[str], chunks: int, out_dir: Path, prefix: str) -> list[tuple[str, Path]]:
    if not items:
        return []
    chunk_size = int(math.ceil(len(items) / chunks))
    paths: list[tuple[str, Path]] = []
    for i in range(chunks):
        part = items[i * chunk_size : (i + 1) * chunk_size]
        if not part:
            break
        out = out_dir / f"tasks_{prefix}_{i:02d}.txt"
        out.write_text("\n".join(part) + "\n")
        paths.append((f"{prefix}_{i:02d}", out))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--belebele-chunks", type=int, default=1)
    parser.add_argument("--flores-chunks", type=int, default=4)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.belebele_chunks < 1 or args.flores_chunks < 1:
        raise SystemExit("belebele/flores chunks must be >= 1")

    tasks_file = Path(args.tasks_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    lines = _read_tasks(tasks_file)
    belebele = [l for l in lines if l.startswith("belebele_")]
    flores = [l for l in lines if l.startswith("flores_en_perplexity_")]

    manifest_lines: list[str] = []
    for tag, path in _write_chunks(belebele, args.belebele_chunks, out_dir, "belebele"):
        manifest_lines.append(f"{tag}\t{path}")
    for tag, path in _write_chunks(flores, args.flores_chunks, out_dir, "flores"):
        manifest_lines.append(f"{tag}\t{path}")

    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
