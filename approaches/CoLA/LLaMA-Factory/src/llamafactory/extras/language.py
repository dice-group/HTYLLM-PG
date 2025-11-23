# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

LANGUAGE_PAD_ID = -1


def load_language_map(spec: Optional[str]) -> Optional[Dict[str, str]]:
    r"""
    Loads a language->family mapping from either an inline JSON string or a file path.
    """
    if spec is None:
        return None

    path = Path(spec)
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(spec)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse language_map '{spec}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("language_map must decode to a dict mapping language -> family.")

    normalized: Dict[str, str] = {}
    for lang, family in data.items():
        if lang is None or family is None:
            continue
        normalized[str(lang)] = str(family)
    return normalized


def build_language_vocab(language_map: Dict[str, str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    r"""
    Builds deterministic vocabularies for languages and families based on the provided mapping.
    """
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
    r"""
    Converts a raw language value to (language_id, family_id) integers.
    Returns -1 for missing or unknown entries so downstream code can ignore them.
    """
    if language_value is None:
        return LANGUAGE_PAD_ID, LANGUAGE_PAD_ID

    lang = str(language_value)
    lang_id = language_vocab.get(lang, LANGUAGE_PAD_ID)
    family = language_map.get(lang)
    family_id = family_vocab.get(family, LANGUAGE_PAD_ID) if family is not None else LANGUAGE_PAD_ID
    return lang_id, family_id
