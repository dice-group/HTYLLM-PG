from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional


@dataclass(frozen=True)
class ColaVariant:
    label: str
    use_experts: bool
    num_A: int
    num_B: int
    strategy: str
    router_mode: str
    prior_weight: float
    bias_value: float
    head_router_mode: str
    head_bias_value: float
    top_k: int
    guidance_scope: str
    train_bs: Optional[int]


@dataclass(frozen=True)
class HydraVariant:
    label: str
    use_experts: bool
    lora_num: int
    router_mode: str
    prior_weight: float
    bias_value: float
    head_router_mode: str
    head_bias_value: float
    top_k: int
    guidance_scope: str
    train_bs: Optional[int]


@dataclass(frozen=True)
class LoraVariant:
    label: str
    train_bs: Optional[int]


@dataclass(frozen=True)
class LanguageTier:
    tier_id: str
    language_count: int
    language_map: str
    tokenized_path: str
    model_path: str


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _parse_int(value: str) -> int:
    return int(value.strip())


def _parse_float(value: str) -> float:
    return float(value.strip())


def _parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if value == "":
        return None
    return int(value)


def _extract_block(lines: Iterable[str], name: str) -> list[str]:
    in_block = False
    block: list[str] = []
    opener = f"{name}=("
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(opener):
            in_block = True
            continue
        if in_block and stripped.startswith(")"):
            break
        if in_block:
            block.append(line.rstrip("\n"))
    return block


def _extract_quoted_value(line: str) -> Optional[str]:
    match = re.search(r'["\\\']([^"\\\']+)["\\\']', line)
    if not match:
        return None
    return match.group(1)


def _parse_variant_lines(lines: Iterable[str], include_commented: bool) -> list[str]:
    variants: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_commented = stripped.startswith("#")
        if is_commented and not include_commented:
            continue
        raw = stripped.lstrip("#").strip()
        value = _extract_quoted_value(raw)
        if value:
            variants.append(value)
    return variants


def parse_cola_variants(script_path: Path, include_commented: bool = True) -> list[ColaVariant]:
    lines = script_path.read_text(encoding="utf-8").splitlines()
    block = _extract_block(lines, "COLA_VARIANTS")
    variants = _parse_variant_lines(block, include_commented)
    parsed: list[ColaVariant] = []
    for spec in variants:
        parts = spec.split("|")
        if len(parts) != 13:
            raise ValueError(f"Unexpected CoLA variant format: {spec}")
        (
            label,
            use_experts,
            num_A,
            num_B,
            strategy,
            router_mode,
            prior_weight,
            bias_value,
            head_router_mode,
            head_bias_value,
            top_k,
            guidance_scope,
            train_bs,
        ) = parts
        parsed.append(
            ColaVariant(
                label=label,
                use_experts=_parse_bool(use_experts),
                num_A=_parse_int(num_A),
                num_B=_parse_int(num_B),
                strategy=strategy,
                router_mode=router_mode,
                prior_weight=_parse_float(prior_weight),
                bias_value=_parse_float(bias_value),
                head_router_mode=head_router_mode,
                head_bias_value=_parse_float(head_bias_value),
                top_k=_parse_int(top_k),
                guidance_scope=guidance_scope,
                train_bs=_parse_optional_int(train_bs),
            )
        )
    return parsed


def parse_hydra_variants(script_path: Path, include_commented: bool = True) -> list[HydraVariant]:
    lines = script_path.read_text(encoding="utf-8").splitlines()
    block = _extract_block(lines, "HYDRA_VARIANTS")
    variants = _parse_variant_lines(block, include_commented)
    parsed: list[HydraVariant] = []
    for spec in variants:
        parts = spec.split("|")
        if len(parts) != 11:
            raise ValueError(f"Unexpected Hydra variant format: {spec}")
        (
            label,
            use_experts,
            lora_num,
            router_mode,
            prior_weight,
            bias_value,
            head_router_mode,
            head_bias_value,
            top_k,
            guidance_scope,
            train_bs,
        ) = parts
        parsed.append(
            HydraVariant(
                label=label,
                use_experts=_parse_bool(use_experts),
                lora_num=_parse_int(lora_num),
                router_mode=router_mode,
                prior_weight=_parse_float(prior_weight),
                bias_value=_parse_float(bias_value),
                head_router_mode=head_router_mode,
                head_bias_value=_parse_float(head_bias_value),
                top_k=_parse_int(top_k),
                guidance_scope=guidance_scope,
                train_bs=_parse_optional_int(train_bs),
            )
        )
    return parsed


def parse_lora_variants(script_path: Path, include_commented: bool = True) -> list[LoraVariant]:
    lines = script_path.read_text(encoding="utf-8").splitlines()
    block = _extract_block(lines, "LORA_VARIANTS")
    variants = _parse_variant_lines(block, include_commented)
    parsed: list[LoraVariant] = []
    for spec in variants:
        parts = spec.split("|")
        if len(parts) != 2:
            raise ValueError(f"Unexpected LoRA variant format: {spec}")
        label, train_bs = parts
        parsed.append(LoraVariant(label=label, train_bs=_parse_optional_int(train_bs)))
    return parsed


def parse_language_tiers(script_path: Path, include_commented: bool = False) -> list[LanguageTier]:
    lines = script_path.read_text(encoding="utf-8").splitlines()
    block = _extract_block(lines, "LANGUAGE_TIERS")
    variants = _parse_variant_lines(block, include_commented)
    parsed: list[LanguageTier] = []
    for spec in variants:
        parts = spec.split("|")
        if len(parts) != 5:
            raise ValueError(f"Unexpected LANGUAGE_TIERS format: {spec}")
        tier_id, language_count, language_map, tokenized_path, model_path = parts
        parsed.append(
            LanguageTier(
                tier_id=tier_id,
                language_count=_parse_int(language_count),
                language_map=language_map,
                tokenized_path=tokenized_path,
                model_path=model_path,
            )
        )
    return parsed


def parse_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    heading_re = re.compile(r"^###\\s+([A-Za-z0-9]+)\\b")
    token_re = re.compile(r"`([^`]+)`")
    variants: dict[str, dict[str, str]] = {}
    current: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = heading_re.match(line)
        if heading:
            current = heading.group(1)
            variants.setdefault(current, {})
            continue
        if current is None:
            continue
        for token in token_re.findall(line):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in variants[current] and variants[current][key] != value:
                raise ValueError(
                    f"Conflicting values for {current}:{key} ({variants[current][key]} vs {value})"
                )
            variants[current][key] = value
    return variants


def default_ground_truth_path() -> Path:
    return Path(__file__).resolve().parents[2] / "papers" / "further" / "acl_multilingual_routing_ground_truth.md"


def default_ablation_script_path() -> Path:
    return Path(__file__).resolve().parent / "run_multilingual_ablation.sh"
