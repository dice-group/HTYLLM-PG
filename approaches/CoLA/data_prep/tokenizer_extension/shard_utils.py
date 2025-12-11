from pathlib import Path
from typing import Iterable, List, Tuple


def find_language_shards(root: Path) -> List[Tuple[str, Path]]:
    """
    Returns (language, file_path) pairs for tokenizer shards.
    Supports two layouts:
      1. root/<language>.jsonl.gz
      2. root/<language>/<anything>.jsonl.gz (first match per directory)
    """
    shards: List[Tuple[str, Path]] = []
    for entry in sorted(_iter_children(root)):
        if entry.is_file():
            language = _strip_suffix(entry.name)
            if language:
                shards.append((language, entry))
        elif entry.is_dir():
            nested = sorted(child for child in entry.iterdir() if _is_jsonl_gz(child))
            if not nested:
                continue
            shards.append((entry.name, nested[0]))
    return shards


def _iter_children(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return list(path.iterdir())


def _strip_suffix(filename: str) -> str:
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def _is_jsonl_gz(path: Path) -> bool:
    return path.is_file() and path.name.endswith(".jsonl.gz")
