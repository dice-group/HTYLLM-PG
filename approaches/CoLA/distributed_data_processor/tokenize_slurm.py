import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset
from transformers import AutoTokenizer

from language_subsets import LANGUAGE_SUBSET_MAP
from distributed_data_processor.validation import ValidationError, verify_tokenized_dataset

LANGUAGE_PAD_ID = -1


def load_language_map(spec: Optional[str]) -> Optional[Dict[str, str]]:
    if spec is None:
        return None

    path = Path(spec)
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = json.loads(spec)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse language_map '{spec}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("language_map must decode to a dict mapping language -> family.")

    normalized: Dict[str, str] = {}
    if all(isinstance(value, str) or value is None for value in data.values()):
        for lang, family in data.items():
            if lang is None or family is None:
                continue
            normalized[str(lang)] = str(family)
        return normalized if normalized else None

    flattened = _flatten_groupings_payload(data)
    if flattened:
        return flattened

    raise ValueError("language_map must decode to language->family or groupings JSON.")


def _flatten_groupings_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for group_id, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("group") or group_id)
        languages = set(entry.get("languages") or entry.get("language") or [])
        subgroups = entry.get("subgroups") or {}
        if isinstance(subgroups, dict):
            for members in subgroups.values():
                if isinstance(members, list):
                    languages.update(members)
        metadata = entry.get("metadata") or {}
        if isinstance(metadata, dict):
            languages.update(metadata.keys())
        for lang in languages:
            lang_key = str(lang)
            normalized.setdefault(lang_key, label)
    return normalized


def build_language_vocab(language_map: Dict[str, str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    languages = sorted(language_map.keys())
    families = sorted(set(language_map.values()))
    language_vocab = {lang: idx for idx, lang in enumerate(languages)}
    family_vocab = {fam: idx for idx, fam in enumerate(families)}
    return language_vocab, family_vocab


def language_to_ids(
    language_value: Optional[str],
    language_map: Dict[str, str],
    language_vocab: Dict[str, int],
    family_vocab: Dict[str, int],
) -> Tuple[int, int]:
    if language_value is None:
        return LANGUAGE_PAD_ID, LANGUAGE_PAD_ID

    lang = str(language_value)
    lang_id = language_vocab.get(lang, LANGUAGE_PAD_ID)
    family = language_map.get(lang)
    family_id = family_vocab.get(family, LANGUAGE_PAD_ID) if family is not None else LANGUAGE_PAD_ID
    return lang_id, family_id


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


LanguageMetadata = Optional[Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]]


def _resolve_language_metadata(language_map_spec: Optional[str]) -> LanguageMetadata:
    if language_map_spec is None:
        return None
    language_map = load_language_map(language_map_spec)
    if language_map is None:
        raise RuntimeError(f"language_map {language_map_spec} contains no languages.")
    language_vocab, family_vocab = build_language_vocab(language_map)
    return language_map, language_vocab, family_vocab


def _language_id_columns(
    languages: List[Optional[str]], language_metadata: LanguageMetadata
) -> Tuple[List[int], List[int]]:
    length = len(languages)
    pad = LANGUAGE_PAD_ID
    if language_metadata is None:
        return [pad] * length, [pad] * length
    language_map, language_vocab, family_vocab = language_metadata
    language_ids: List[int] = []
    family_ids: List[int] = []
    for language in languages:
        lang_id, fam_id = language_to_ids(language, language_map, language_vocab, family_vocab)
        language_ids.append(lang_id)
        family_ids.append(fam_id)
    return language_ids, family_ids


def tokenize_fn(batch, tokenizer, keep_text: bool, language_metadata: LanguageMetadata):
    tokenized = tokenizer(batch["text"], truncation=True, padding=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    languages = batch.get("language")
    if languages is None:
        languages = [None] * len(batch["text"])
    language_ids, family_ids = _language_id_columns(languages, language_metadata)
    tokenized["language_ids"] = language_ids
    tokenized["family_ids"] = family_ids
    if keep_text:
        tokenized["text"] = batch["text"]
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

    language_metadata = _resolve_language_metadata(getattr(args, "language_map", None))

    with tempfile.TemporaryDirectory(prefix="tok_simple_") as cache_dir:
        raw_dataset = Dataset.from_generator(
            lambda: _iter_shard_lines(assigned, args.eval_fraction, args.eval_seed),
            cache_dir=cache_dir,
        )

        print(f"Rank {rank}: start tokenizing {len(assigned)} part(s)")
        tokenized_dataset = raw_dataset.map(
            lambda batch: tokenize_fn(
                batch, tokenizer, getattr(args, "keep_text", False), language_metadata
            ),
            batched=True,
            remove_columns=None if getattr(args, "keep_text", False) else ["text"],
            num_proc=args.num_proc,
        )

        try:
            verify_tokenized_dataset(
                tokenized_dataset,
                require_language_metadata=language_metadata is not None,
            )
        except ValidationError as exc:
            raise RuntimeError("Tokenized dataset validation failed") from exc

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
    parser.add_argument("--keep_text", action="store_true", help="Keep the raw text column in the tokenized dataset.")
    parser.add_argument("--language-map", type=str, default=None, help="Optional path or JSON string mapping languages to families.")
    args = parser.parse_args()
    main(args)
