from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtensionConfig:
    base_tokenizer_path: Path
    multilingual_tokenizer_path: Path
    data_dir: Path
    output_dir: Path
    sample_docs: int = 50
    text_key: str = "text"
    vocab_cap: Optional[int] = 256_000
    num_workers: Optional[int] = None
    init_embeddings: bool = False
    model_path: Optional[str] = None
    initialized_model_dir: Optional[Path] = None


@dataclass
class ExtensionResult:
    added_tokens: int
    total_vocab_size: int
    per_language: Dict[str, int]
    initialized_model_dir: Optional[Path] = None


def extend_tokenizer(
    config: ExtensionConfig,
    allocation,
) -> ExtensionResult:
    base_tokenizer = AutoTokenizer.from_pretrained(str(config.base_tokenizer_path), use_fast=True)
    multilingual_tokenizer = AutoTokenizer.from_pretrained(
        str(config.multilingual_tokenizer_path), use_fast=True
    )

    base_vocab = set(base_tokenizer.get_vocab().keys())
    global_tokens: List[str] = []
    per_language: Dict[str, int] = {}

    tasks: List[Tuple[int, str, int]] = []
    for order, row in enumerate(allocation.itertuples(index=False)):
        language = getattr(row, "language")
        n_tokens = int(getattr(row, "token_alloc"))
        if language == "english":
            print("[tokenize_extension] Skipping English, extension disabled.")
            continue
        tasks.append((order, language, n_tokens))

    def process_language(language: str, n_tokens: int) -> Optional[List[str]]:
        lang_file = config.data_dir / f"{language}.jsonl.gz"
        if not lang_file.exists():
            print(f"[tokenize_extension] Skipping {language}, missing file {lang_file}")
            return None

        texts = _load_texts(lang_file, config.sample_docs, config.text_key)
        if not texts:
            print(f"[tokenize_extension] Skipping {language}, no texts loaded.")
            return None

        counts = Counter()
        for text in texts:
            tokens = multilingual_tokenizer.tokenize(text)
            counts.update(tok for tok in tokens if tok not in base_vocab)

        candidates = [tok for tok, _ in counts.most_common() if tok not in base_vocab]
        selected = candidates[:n_tokens]
        print(f"[tokenize_extension] {language}: requested {n_tokens}, added {len(selected)} tokens.")
        return selected

    def determine_worker_count(total_tasks: int) -> int:
        if total_tasks <= 1:
            return 1
        if config.num_workers is not None:
            return max(1, min(config.num_workers, total_tasks))
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, total_tasks))

    results: Dict[str, Optional[List[str]]] = {}
    if tasks:
        worker_count = determine_worker_count(len(tasks))
        if worker_count == 1:
            for _, language, n_tokens in tasks:
                results[language] = process_language(language, n_tokens)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(process_language, language, n_tokens): language
                    for _, language, n_tokens in tasks
                }
                for future in as_completed(future_map):
                    language = future_map[future]
                    results[language] = future.result()

    for _, language, _ in sorted(tasks):
        selected = results.get(language)
        if selected is None:
            continue
        per_language[language] = len(selected)
        if selected:
            global_tokens.extend(selected)

    tokens_added = _merge_and_save(
        config.base_tokenizer_path,
        config.multilingual_tokenizer_path,
        config.output_dir,
        global_tokens,
        config.vocab_cap,
    )
    total_size = len(AutoTokenizer.from_pretrained(str(config.output_dir), use_fast=True))
    initialized_model_dir = _maybe_initialize_model_embeddings(config)

    return ExtensionResult(
        added_tokens=tokens_added,
        total_vocab_size=total_size,
        per_language=per_language,
        initialized_model_dir=initialized_model_dir,
    )


def _load_texts(path: Path, limit: int, text_key: str) -> List[str]:
    texts: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit and idx >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get(text_key)
            if text:
                texts.append(text)
    return texts


def _merge_and_save(
    base_path: Path,
    multilingual_path: Path,
    output_dir: Path,
    selected_tokens: List[str],
    vocab_cap: Optional[int],
) -> int:
    base_json_path = base_path / "tokenizer.json"
    multi_json_path = multilingual_path / "tokenizer.json"

    if not base_json_path.exists() or not multi_json_path.exists():
        raise FileNotFoundError("tokenizer.json not found for base or multilingual tokenizer.")

    base_data = json.loads(base_json_path.read_text(encoding="utf-8"))
    multi_data = json.loads(multi_json_path.read_text(encoding="utf-8"))

    base_vocab_map = base_data["model"]["vocab"]
    base_merges = _normalize_merges(base_data["model"]["merges"])
    multi_merges = _normalize_merges(multi_data["model"]["merges"])

    base_vocab_set = set(base_vocab_map.keys())
    ordered_tokens = []
    seen = set()
    for tok in selected_tokens:
        if tok not in base_vocab_set and tok not in seen:
            ordered_tokens.append(tok)
            seen.add(tok)

    merge_tokens = _extract_merge_tokens(multi_merges)
    for tok in merge_tokens:
        if tok not in base_vocab_set and tok not in seen:
            ordered_tokens.append(tok)
            seen.add(tok)

    if vocab_cap is not None:
        max_new = max(vocab_cap - len(base_vocab_map), 0)
        if len(ordered_tokens) > max_new:
            ordered_tokens = ordered_tokens[:max_new]

    vocab = dict(base_vocab_map)
    next_id = max(vocab.values()) + 1
    added = 0
    for tok in ordered_tokens:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
            added += 1

    new_merges = [m for m in multi_merges if m not in set(base_merges)]
    valid_merges = [
        m
        for m in new_merges
        if _merge_parts_exist(m, vocab)
    ]

    base_data["model"]["vocab"] = vocab
    base_data["model"]["merges"] = base_merges + valid_merges

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer.json").write_text(
        json.dumps(base_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for filename in ("tokenizer_config.json", "special_tokens_map.json"):
        src = base_path / filename
        if src.exists():
            (output_dir / filename).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(
        f"[tokenize_extension] Saved extended tokenizer to {output_dir} "
        f"(tokens added: {added}, merges added: {len(valid_merges)})"
    )
    return added


def _normalize_merges(merges) -> List[str]:
    normalized = []
    for item in merges:
        if isinstance(item, str):
            normalized.append(item)
        else:
            normalized.append(" ".join(item))
    return normalized


def _extract_merge_tokens(merges: List[str]) -> List[str]:
    parts = set()
    results = set()
    for merge in merges:
        tokens = merge.split()
        if len(tokens) == 2:
            parts.update(tokens)
            results.add("".join(tokens))
    return sorted(parts | results)


def _merge_parts_exist(merge: str, vocab: Dict[str, int]) -> bool:
    tokens = merge.split()
    if len(tokens) != 2:
        return False
    left, right = tokens
    result = "".join(tokens)
    return left in vocab and right in vocab and result in vocab


def _maybe_initialize_model_embeddings(config: ExtensionConfig) -> Optional[Path]:
    if not config.init_embeddings:
        return None
    if not config.model_path:
        print("[tokenize_extension] Skipping embedding initialization, model_path not provided.")
        return None

    target_dir = config.initialized_model_dir or (config.output_dir.parent / "initialized_model")
    target_dir.mkdir(parents=True, exist_ok=True)

    base_tokenizer = AutoTokenizer.from_pretrained(str(config.base_tokenizer_path), use_fast=True)
    extended_tokenizer = AutoTokenizer.from_pretrained(str(config.output_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(config.model_path))

    new_vocab_size = len(extended_tokenizer)
    embedding_layer = model.get_input_embeddings()
    original_vocab_size = embedding_layer.weight.shape[0]

    if new_vocab_size != original_vocab_size:
        model.resize_token_embeddings(new_vocab_size)
        embedding_layer = model.get_input_embeddings()

    output_layer = model.get_output_embeddings()

    base_vocab = base_tokenizer.get_vocab()
    extended_vocab = extended_tokenizer.get_vocab()
    new_tokens = [tok for tok in extended_vocab if tok not in base_vocab]

    print(f"[tokenize_extension] Initializing embeddings for {len(new_tokens)} new tokens.")
    with torch.no_grad():
        for token in new_tokens:
            token_id = extended_vocab[token]
            text = extended_tokenizer.convert_tokens_to_string([token])
            if not text:
                continue
            base_ids = base_tokenizer(text, add_special_tokens=False)["input_ids"]
            valid_ids = [idx for idx in base_ids if 0 <= idx < original_vocab_size]
            if not valid_ids:
                continue
            mean_vec = embedding_layer.weight[valid_ids].mean(dim=0)
            embedding_layer.weight[token_id] = mean_vec
            if output_layer is not None and output_layer.weight.shape[0] == new_vocab_size:
                output_layer.weight[token_id] = mean_vec

    model.save_pretrained(target_dir)
    extended_tokenizer.save_pretrained(target_dir / "tokenizer")
    print(f"[tokenize_extension] Saved initialized model to {target_dir}")
    return target_dir