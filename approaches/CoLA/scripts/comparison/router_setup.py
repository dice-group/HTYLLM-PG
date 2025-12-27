from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import json


@dataclass
class RouterConfig:
    kind: str
    tier: Optional[str]
    variant: Optional[str]
    use_experts: bool
    num_experts: Optional[int]
    router_mode: Optional[str]
    head_router_mode: Optional[str]
    guidance_scope: Optional[str]
    language_map: Optional[str]
    prior_weight: Optional[float]
    bias_value: Optional[float]
    head_bias_value: Optional[float]
    top_k: Optional[int]
    lora_num: Optional[int]
    expected_experts: Optional[int]
    expected_heads: Optional[list[int]]
    expert_heads_override: Optional[list[int]]


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
    return None


def _parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None or value.strip() == "":
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None
    values = []
    for item in items:
        try:
            values.append(int(item))
        except ValueError:
            return None
    return values


def _read_language_groupings(path: str) -> tuple[Optional[int], Optional[list[int]]]:
    payload = Path(path)
    try:
        if payload.exists():
            data = json.loads(payload.read_text(encoding="utf-8"))
        else:
            data = json.loads(path)
    except (OSError, json.JSONDecodeError):
        return None, None

    if not isinstance(data, dict):
        return None, None

    if all(isinstance(value, str) or value is None for value in data.values()):
        return len(set(data.values())), None

    expert_count = 0
    head_counts: list[int] = []
    for _, entry in sorted(data.items(), key=lambda kv: str(kv[0])):
        if not isinstance(entry, dict):
            continue
        expert_count += 1
        subgroups = entry.get("subgroups") or {}
        if isinstance(subgroups, dict):
            head_counts.append(len(subgroups))
        else:
            head_counts.append(0)
    return expert_count, head_counts


def build_router_config(kind: str, env: Mapping[str, str]) -> RouterConfig:
    kind = kind.lower()
    if kind not in {"cola", "hydra"}:
        raise ValueError(f"Unknown router kind {kind}")

    if kind == "hydra":
        use_experts = _parse_bool(env.get("USE_HYDRALORA_EXPERTS"))
        num_experts = _parse_int(env.get("HYDRALORA_NUM_EXPERTS"))
        top_k = _parse_int(env.get("HYDRALORA_TOP_K"))
        lora_num = _parse_int(env.get("LORA_NUM"))
        expert_heads_override = _parse_int_list(env.get("HYDRALORA_EXPERT_LORA_NUMS"))
    else:
        use_experts = _parse_bool(env.get("USE_COLA_EXPERTS"))
        num_experts = _parse_int(env.get("COLA_NUM_EXPERTS"))
        top_k = _parse_int(env.get("COLA_TOP_K"))
        lora_num = _parse_int(env.get("COLA_NUM_B"))
        expert_heads_override = _parse_int_list(env.get("COLA_EXPERT_NUM_B"))

    language_map = env.get("LANGUAGE_MAP")
    expected_experts = None
    expected_heads = None
    if language_map:
        expected_experts, expected_heads = _read_language_groupings(language_map)

    return RouterConfig(
        kind=kind,
        tier=env.get("LANGUAGE_TIER"),
        variant=env.get("MODEL_VARIANT"),
        use_experts=use_experts,
        num_experts=num_experts,
        router_mode=env.get("LANGUAGE_ROUTER_MODE"),
        head_router_mode=env.get("LANGUAGE_HEAD_ROUTER_MODE"),
        guidance_scope=env.get("LANGUAGE_GUIDANCE_SCOPE"),
        language_map=language_map,
        prior_weight=_parse_float(env.get("LANGUAGE_PRIOR_WEIGHT")),
        bias_value=_parse_float(env.get("LANGUAGE_BIAS_VALUE")),
        head_bias_value=_parse_float(env.get("LANGUAGE_HEAD_BIAS_VALUE")),
        top_k=top_k,
        lora_num=lora_num,
        expected_experts=expected_experts,
        expected_heads=expected_heads,
        expert_heads_override=expert_heads_override,
    )


def describe_router_config(config: RouterConfig) -> None:
    expected_heads = ",".join(str(v) for v in config.expected_heads) if config.expected_heads else "unknown"
    override_heads = (
        ",".join(str(v) for v in config.expert_heads_override)
        if config.expert_heads_override
        else "none"
    )
    parts = [
        f"[router-setup] kind={config.kind}",
        f"tier={config.tier or 'unknown'}",
        f"variant={config.variant or 'unspecified'}",
        f"use_experts={config.use_experts}",
        f"num_experts={config.num_experts or 0}",
        f"expected_experts={config.expected_experts or 0}",
        f"expected_heads={expected_heads}",
        f"override_heads={override_heads}",
        f"num_B_or_heads={config.lora_num or 0}",
        f"router_mode={config.router_mode or 'default'}",
        f"head_router_mode={config.head_router_mode or 'default'}",
        f"guidance_scope={config.guidance_scope or 'default'}",
        f"language_map={config.language_map or 'NONE'}",
        f"prior_weight={config.prior_weight or 0:.3f}",
        f"head_bias_value={config.head_bias_value or 0:.3f}",
        f"top_k={config.top_k or 0}",
    ]
    print(" ".join(parts))


def validate_router_config(config: RouterConfig) -> None:
    if not config.language_map:
        raise ValueError("LANGUAGE_MAP is required for router-based models.")
    if not Path(config.language_map).exists():
        raise ValueError(f"language_map path {config.language_map} does not exist.")

    if config.use_experts and (config.num_experts is None or config.num_experts <= 0):
        raise ValueError("Expert routing enabled but COLA/Hydra num experts is not positive.")

    if config.use_experts and config.expected_experts is not None:
        if config.num_experts != config.expected_experts:
            raise ValueError(
                "Expert count mismatch: "
                f"expected {config.expected_experts}, got {config.num_experts}."
            )

    if config.use_experts and config.expected_heads:
        expected_heads = config.expected_heads
        if config.expert_heads_override:
            if config.expert_heads_override != expected_heads:
                raise ValueError(
                    "Expert head count mismatch vs override: "
                    f"expected {expected_heads}, got {config.expert_heads_override}."
                )
        elif config.lora_num is not None and len(set(expected_heads)) == 1:
            if config.lora_num != expected_heads[0]:
                raise ValueError(
                    "Expert head count mismatch vs num_B/lora_num: "
                    f"expected {expected_heads[0]}, got {config.lora_num}."
                )

    if config.kind == "hydra" and config.lora_num is None:
        raise ValueError("LORA_NUM must be set for Hydra variants.")

    if config.guidance_scope not in {None, "all", "expert_only", "none"}:
        raise ValueError(f"Unexpected guidance scope: {config.guidance_scope}")

    if config.router_mode not in {None, "learned", "bias", "hard"}:
        raise ValueError(f"Router mode {config.router_mode} is not supported.")

    if config.head_router_mode not in {None, "", "learned", "bias", "hard"}:
        raise ValueError(f"Head router mode {config.head_router_mode} is not supported.")

    print(f"[router-setup] validation passed for {config.kind} variant.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log and validate router configurations.")
    parser.add_argument("--type", choices=["hydra", "cola"], required=True)
    args = parser.parse_args()

    config = build_router_config(args.type, os.environ)
    describe_router_config(config)
    validate_router_config(config)


if __name__ == "__main__":
    main()
