import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from datasets import Dataset
from transformers import AutoTokenizer

from language_subsets import (
    five_representatives_mediods,
    fourty_six_representatives_mediods,
    hundred_ninty_nine_representatives_mediods,
    ninty_five_representatives_mediods,
    ten_representatives_mediods,
    twenty_two_representatives_mediods,
)

LANGUAGE_SUBSET_MAP = {
    "five_representatives_mediods": five_representatives_mediods,
    "ten_representatives_mediods": ten_representatives_mediods,
    "twenty_two_representatives_mediods": twenty_two_representatives_mediods,
    "fourty_six_representatives_mediods": fourty_six_representatives_mediods,
    "ninty_five_representatives_mediods": ninty_five_representatives_mediods,
    "hundred_ninty_nine_representatives_mediods": hundred_ninty_nine_representatives_mediods,
}


def _resolve_rank_world(rank: Optional[int], world_size: Optional[int]) -> tuple[int, int]:
    if rank is not None:
        resolved_rank = rank
    else:
        env_rank = (
            os.getenv("SLURM_ARRAY_TASK_ID")
            or os.getenv("SLURM_PROCID")
            or os.getenv("RANK")
            or os.getenv("LOCAL_RANK")
            or 0
        )
        resolved_rank = int(env_rank)

    if world_size is not None:
        resolved_world = world_size
    else:
        env_world = (
            os.getenv("SLURM_ARRAY_TASK_COUNT")
            or os.getenv("SLURM_NTASKS")
            or os.getenv("WORLD_SIZE")
            or 1
        )
        resolved_world = int(env_world)

    return resolved_rank, max(1, resolved_world)


def _list_shards(shard_dir: Path, languages: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    valid_suffixes = (".jsonl", ".jsonl.gz")
    search_roots = [shard_dir] if not languages else [shard_dir / lang for lang in languages]
    shards: List[Tuple[str, Path]] = []
    for root in search_roots:
        if not root.exists():
            raise RuntimeError(f"Requested language directory is missing: {root}")
        for path in root.rglob("*"):
            if not path.is_file() or not any(path.name.endswith(sfx) for sfx in valid_suffixes):
                continue
            rel = path.relative_to(shard_dir)
            lang = rel.parts[0] if rel.parts else path.parent.name
            shards.append((lang, path))
    shards.sort(key=lambda item: item[1])
    if not shards:
        raise RuntimeError(f"No .jsonl or .jsonl.gz files found under {shard_dir}")
    return shards


def _assign_shards(shards: List[Tuple[str, Path]], rank: int, world_size: int) -> List[Tuple[str, Path]]:
    if world_size <= 1:
        return shards
    assigned = [item for idx, item in enumerate(shards) if idx % world_size == rank]
    print(f"Rank {rank}: assigned {len(assigned)} part files out of {len(shards)}")
    return assigned


def _iter_shard_lines(
    shards: List[Tuple[str, Path]],
    eval_fraction: float,
    eval_seed: int,
) -> Iterator[dict]:
    for language, shard_path in shards:
        opener = gzip.open if shard_path.suffix == ".gz" else open
        with opener(shard_path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                text = _extract_text(line)
                if text is not None:
                    record = {"text": text, "language": language}
                    if eval_fraction > 0:
                        record["split"] = (
                            "validation"
                            if _assign_to_eval(language, text, eval_fraction, eval_seed)
                            else "train"
                        )
                    yield record


def _extract_text(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                return text
    except json.JSONDecodeError:
        pass
    return stripped


def tokenize_fn(batch, tokenizer):
    tokenized = tokenizer(batch["text"], truncation=True, padding=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def _resolve_languages(explicit: Optional[List[str]], subset_name: Optional[str]) -> Optional[List[str]]:
    if explicit and subset_name:
        raise RuntimeError("Use either --languages or --language_subset, not both.")
    if subset_name:
        if subset_name not in LANGUAGE_SUBSET_MAP:
            available = ", ".join(sorted(LANGUAGE_SUBSET_MAP))
            raise RuntimeError(f"{subset_name} not found. Available subsets: {available}")
        return LANGUAGE_SUBSET_MAP[subset_name]
    return explicit


def _assign_to_eval(language: str, text: str, fraction: float, seed: int) -> bool:
    key = f"{language}\u0000{seed}\u0000{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def main(args):
    shard_dir = Path(args.shard_dir)
    print(f"Tokenizer used for tokenization: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    rank, world_size = _resolve_rank_world(args.rank, args.world_size)
    languages = _resolve_languages(args.languages, args.language_subset)
    shards = _list_shards(shard_dir, languages)
    assigned = _assign_shards(shards, rank, world_size)
    if not assigned:
        raise RuntimeError(f"Rank {rank} received zero shard files. Check shard count vs. world size.")

    with tempfile.TemporaryDirectory(prefix="tok_simple_") as cache_dir:
        raw_dataset = Dataset.from_generator(
            lambda: _iter_shard_lines(assigned, args.eval_fraction, args.eval_seed),
            cache_dir=cache_dir,
        )

        print(f"Rank {rank}: start tokenizing {len(assigned)} part(s)")
        tokenized_dataset = raw_dataset.map(
            lambda batch: tokenize_fn(batch, tokenizer),
            batched=True,
            remove_columns=["text"],
            num_proc=args.num_proc,
        )

        output_dir = Path(args.save_tokenized_data_dir)
        if world_size > 1:
            output_dir = output_dir / f"rank_{rank:05d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenized_dataset.save_to_disk(str(output_dir))
        print(f"Rank {rank}: tokenized data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize pre-split corpus parts.")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing <lang>_part_*.jsonl(.gz) files.")
    parser.add_argument("--save_tokenized_data_dir", type=str, required=True, help="Directory to save tokenized shards.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Tokenizer model name or path.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of workers for tokenizer.map.")
    parser.add_argument("--rank", type=int, default=None, help="Rank override (defaults to SLURM env vars).")
    parser.add_argument("--languages", nargs="+", default=None, help="Optional list of language directory names (e.g., eng_Latn deu_Latn).")
    parser.add_argument("--language_subset", type=str, default=None, help="Name of a language list defined in language_subsets.py (e.g., twenty_two_representatives_mediods).")
    parser.add_argument("--world_size", type=int, default=None, help="World size override (defaults to SLURM env vars).")
    parser.add_argument("--eval_fraction", type=float, default=0.05, help="Fraction per language to tag as validation data (set 0 to disable).")
    parser.add_argument("--eval_seed", type=int, default=42, help="Seed for the eval split hash.")
    args = parser.parse_args()
    main(args)
