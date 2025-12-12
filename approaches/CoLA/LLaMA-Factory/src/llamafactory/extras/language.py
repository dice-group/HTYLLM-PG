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
from typing import Any, Dict, Optional, Tuple

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


def load_language_groupings(
    spec: Optional[str],
) -> tuple[Optional[Dict[str, str]], Optional[list[str]], Optional[list[int]], Optional[Dict[str, int]]]:
    """
    Parse a language grouping JSON (tier files) and return:
      - language_map: lang -> group label (family)
      - families: sorted list of group labels
      - subgroup_sizes: list of subgroup counts per family (aligned with `families`)
      - language_to_subgroup: mapping lang -> subgroup index within its family

    Falls back to the flat language_map parsing when the payload is not a grouping.
    """
    if spec is None:
        return None, None, None, None

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
        raise ValueError("language_map must decode to a dict mapping language -> family or grouping entries.")

    if all(isinstance(value, str) or value is None for value in data.values()):
        language_map = load_language_map(spec)
        if language_map is None:
            return None, None, None, None
        families = sorted(set(language_map.values()))
        subgroup_sizes = [0 for _ in families]
        return language_map, families, subgroup_sizes, None

    language_map: Dict[str, str] = {}
    subgroup_sizes: Dict[str, int] = {}
    language_to_subgroup: Dict[str, int] = {}

    for group_id, entry in sorted(data.items(), key=lambda kv: str(kv[0])):
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("group") or group_id)
        languages = set(entry.get("languages") or entry.get("language") or [])
        subgroups = entry.get("subgroups") or {}

        subgroup_count = 0
        if isinstance(subgroups, dict) and subgroups:
            for idx, (_, members) in enumerate(sorted(subgroups.items(), key=lambda kv: kv[0])):
                if not isinstance(members, list):
                    continue
                subgroup_count += 1
                for lang in members:
                    languages.add(lang)
                    language_to_subgroup[str(lang)] = idx
        subgroup_sizes[label] = subgroup_count

        metadata = entry.get("metadata") or {}
        if isinstance(metadata, dict):
            languages.update(metadata.keys())

        for lang in languages:
            lang_key = str(lang)
            language_map.setdefault(lang_key, label)

    if not language_map:
        return None, None, None, None

    families = sorted(set(language_map.values()))
    subgroup_sizes_aligned = [subgroup_sizes.get(fam, 0) for fam in families]
    return language_map, families, subgroup_sizes_aligned, language_to_subgroup


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
