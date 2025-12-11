import argparse
import json
import os
import shutil
import uuid
from pathlib import Path


def discover_rank_dirs(root: Path) -> list[Path]:
    ranks = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("rank_"))
    if not ranks:
        raise RuntimeError("No rank_* dirs found")
    return ranks


def _list_rank_files(rank_dir: Path) -> list[str]:
    state_path = rank_dir / "state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            files = [
                f["filename"]
                for f in state.get("_data_files", [])
                if "filename" in f
            ]
            if files:
                return files
        except Exception:
            pass
    return sorted(p.name for p in rank_dir.glob("data-*.arrow"))


def fast_merge(root: Path, output: Path, overwrite: bool):
    """Symlink rank_* shards into one HF dataset directory."""
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"{output} exists. Use --overwrite.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    ranks = discover_rank_dirs(root)
    print(f"[merge] fast merge {len(ranks)} ranks from {root} -> {output}")
    info = ranks[0] / "dataset_info.json"
    if info.is_file():
        shutil.copy2(info, output / "dataset_info.json")
        print(f"[merge] copied dataset_info.json from {ranks[0].name}")
    data_files = []
    total_files = 0
    for i, rd in enumerate(ranks):
        link = output / rd.name
        if not link.exists():
            os.symlink(os.path.relpath(rd, output), link)
        files = _list_rank_files(rd)
        total_files += len(files)
        if i < 3 or i == len(ranks) - 1:
            print(f"[merge] {rd.name}: {len(files)} shard(s)")
        data_files.extend({"filename": f"{rd.name}/{f}"} for f in files)
    state = {
        "_data_files": data_files,
        "_fingerprint": uuid.uuid4().hex,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    (output / "state.json").write_text(json.dumps(state, indent=2))
    print(f"[merge] wrote state.json with {total_files} arrow shard(s)")
    print(f"[merge] fast dataset -> {output}")


def main():
    ap = argparse.ArgumentParser(description="Metadata-only merge of rank_* HF datasets.")
    ap.add_argument("--tokenized_root", type=Path, required=True)
    ap.add_argument("--output_path", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    fast_merge(args.tokenized_root, args.output_path, args.overwrite)


if __name__ == "__main__":
    main()
