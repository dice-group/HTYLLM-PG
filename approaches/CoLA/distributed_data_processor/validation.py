from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset

LANGUAGE_PAD_ID = -1


class ValidationError(ValueError):
    """Raised when tokenized datasets are malformed or missing metadata."""


def _has_nonpad_ids(ids: Iterable[int]) -> bool:
    return any(id_ >= 0 for id_ in ids)


def verify_tokenized_dataset(
    dataset: Dataset,
    *,
    require_language_metadata: bool = True,
    sample_size: int = 10,
) -> None:
    """
    Ensures the tokenized HuggingFace dataset exposes the columns and metadata the models expect.

    Args:
        dataset: The Arrow dataset produced by `tokenize_slurm.py`.
        require_language_metadata: If True, at least one row must carry non-pad `language_ids`.
        sample_size: How many rows to peek at for the metadata check.
    """
    required_columns = {"input_ids", "attention_mask", "labels", "language_ids", "family_ids"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValidationError(f"Missing columns: {', '.join(sorted(missing))}")

    if dataset.num_rows == 0:
        raise ValidationError("Dataset contains zero rows.")

    rows = min(sample_size, dataset.num_rows)
    idxs = list(range(rows))
    sample = dataset.select(idxs)
    language_ids = sample["language_ids"]
    family_ids = sample["family_ids"]
    if len(language_ids) != rows or len(family_ids) != rows:
        raise ValidationError("Language metadata columns have unexpected lengths.")

    if require_language_metadata and not _has_nonpad_ids(language_ids):
        raise ValidationError("All `language_ids` values are pads; metadata is missing.")

    if require_language_metadata and not _has_nonpad_ids(family_ids):
        raise ValidationError("All `family_ids` values are pads; metadata is missing.")

    print(f"[validation] dataset OK ({dataset.num_rows} rows, sample_size={rows}).")
