from __future__ import annotations

import pytest
from datasets import Dataset

from data_prep.merlin_data_prep.distributed_data_processor.src.validation import (
    ValidationError,
    verify_tokenized_dataset,
)


def _build_dataset(language_id: int, family_id: int, include_family=True) -> Dataset:
    data = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[1, 2, 3]],
        "language_ids": [language_id],
    }
    if include_family:
        data["family_ids"] = [family_id]
    return Dataset.from_dict(data)


def test_verify_tokenized_dataset_succeeds_with_metadata() -> None:
    ds = _build_dataset(language_id=0, family_id=1)
    verify_tokenized_dataset(ds, require_language_metadata=True)


def test_verify_tokenized_dataset_requires_columns() -> None:
    ds = _build_dataset(language_id=0, family_id=1, include_family=False)
    with pytest.raises(ValidationError):
        verify_tokenized_dataset(ds, require_language_metadata=False)


def test_verify_tokenized_dataset_rejects_all_pad_ids() -> None:
    ds = _build_dataset(language_id=-1, family_id=-1)
    with pytest.raises(ValidationError):
        verify_tokenized_dataset(ds, require_language_metadata=True)
