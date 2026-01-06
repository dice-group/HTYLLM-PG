import gzip
import hashlib
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


SHARD_DIR = Path("/data/project_data/moe_study/fw_samples/preview_subset")
OUTPUT_DIR = Path("/data/project_data/moe_study/tokenized/preview_subset_tiny_llama")
MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"
LANGUAGE_MAP_PATH = Path(
    "/upb/users/j/joeldag/profiles/unix/cs/HTYLLM-PG/approaches/CoLA/tools/two_stage_clustering/200_tier_language_groupings.json"
)
EVAL_FRACTION = 0.02
EVAL_SEED = 42
MAX_LENGTH = 1024

LANGUAGE_PAD_ID = -1


def _assign_to_eval(language: str, text: str, fraction: float, seed: int) -> bool:
    key = f"{language}\u0000{seed}\u0000{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def _extract_text(line: str) -> str | None:
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


def _load_language_metadata():
    from llamafactory.extras.language import load_language_groupings

    language_map, families, _, _ = load_language_groupings(str(LANGUAGE_MAP_PATH))
    if not language_map:
        raise RuntimeError(f"language_map is empty: {LANGUAGE_MAP_PATH}")
    language_list = sorted(language_map.keys())
    family_list = sorted(set(language_map.values())) if families is None else families
    language_vocab = {lang: idx for idx, lang in enumerate(language_list)}
    family_vocab = {fam: idx for idx, fam in enumerate(family_list)}
    return language_map, language_vocab, family_vocab


def _language_to_ids(language: str, language_map, language_vocab, family_vocab) -> tuple[int, int]:
    if language is None:
        return LANGUAGE_PAD_ID, LANGUAGE_PAD_ID
    lang_id = language_vocab.get(language, LANGUAGE_PAD_ID)
    family = language_map.get(language)
    fam_id = family_vocab.get(family, LANGUAGE_PAD_ID) if family is not None else LANGUAGE_PAD_ID
    return lang_id, fam_id


def _iter_records():
    for path in sorted(SHARD_DIR.glob("*.jsonl.gz")):
        language = path.name.replace(".jsonl.gz", "")
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                text = _extract_text(line)
                if text is None:
                    continue
                yield language, text


def main() -> None:
    if not SHARD_DIR.exists():
        raise RuntimeError(f"Missing input shard dir: {SHARD_DIR}")
    if not LANGUAGE_MAP_PATH.exists():
        raise RuntimeError(f"Missing language_map: {LANGUAGE_MAP_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    language_map, language_vocab, family_vocab = _load_language_metadata()
    train_rows = []
    valid_rows = []
    skipped = 0

    for language, text in _iter_records():
        if language not in language_map:
            skipped += 1
            continue
        record = {"text": text, "language": language}
        if _assign_to_eval(language, text, EVAL_FRACTION, EVAL_SEED):
            valid_rows.append(record)
        else:
            train_rows.append(record)

    if not train_rows:
        raise RuntimeError("No training rows collected; check input files.")
    if not valid_rows:
        raise RuntimeError("No validation rows collected; check eval split settings.")

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        language_ids = []
        family_ids = []
        for lang in batch["language"]:
            lang_id, fam_id = _language_to_ids(lang, language_map, language_vocab, family_vocab)
            language_ids.append(lang_id)
            family_ids.append(fam_id)
        tokenized["language_ids"] = language_ids
        tokenized["family_ids"] = family_ids
        return tokenized

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)

    train_tok = train_ds.map(tokenize_batch, batched=True, remove_columns=["text", "language"])
    valid_tok = valid_ds.map(tokenize_batch, batched=True, remove_columns=["text", "language"])

    dataset = DatasetDict(train=train_tok, validation=valid_tok)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)

    print(f"Saved tokenized dataset to {OUTPUT_DIR}")
    print(f"Train samples: {len(train_tok)} | Validation samples: {len(valid_tok)}")
    if skipped:
        print(f"Skipped {skipped} samples with unknown languages.")


if __name__ == "__main__":
    main()
