from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

from scripts.comparison.ablation_specs import (
    default_ablation_script_path,
    parse_cola_variants,
    parse_hydra_variants,
    parse_language_tiers,
    parse_lora_variants,
)


def _count_experts_from_language_map(path: str) -> int:
    payload = Path(path)
    with payload.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    raise ValueError(f"Unsupported language_map format: {type(data).__name__}")


def _select(items: Iterable[str], allowed: set[str]) -> list[str]:
    selected = []
    for item in items:
        if item in allowed:
            selected.append(item)
    return selected


def _run_job(script: Path, env: dict[str, str], dry_run: bool) -> None:
    cmd = ["bash", str(script)]
    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        for key in sorted(env.keys()):
            if key in {"REPO_ROOT", "OUTPUT_DIR", "MODEL_NAME_OR_PATH", "TOKENIZED_PATH", "LANGUAGE_MAP"}:
                print(f"  {key}={env[key]}")
        return
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multilingual ablation jobs locally without Slurm.")
    parser.add_argument("--tiers", default="", help="Comma-separated tier ids to run (default: all).")
    parser.add_argument("--only", default="cola,hydra,lora", help="Comma-separated kinds: cola,hydra,lora.")
    parser.add_argument("--include-commented", action="store_true", help="Include commented variants.")
    parser.add_argument("--include-commented-tiers", action="store_true", help="Include commented tiers.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to target.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    script_path = default_ablation_script_path()
    cola_variants = parse_cola_variants(script_path, include_commented=args.include_commented)
    hydra_variants = parse_hydra_variants(script_path, include_commented=args.include_commented)
    lora_variants = parse_lora_variants(script_path, include_commented=args.include_commented)
    tiers = parse_language_tiers(script_path, include_commented=args.include_commented_tiers)

    allowed_kinds = set(_select(args.only.split(","), {"cola", "hydra", "lora"}))
    tier_filter = set(t.strip() for t in args.tiers.split(",") if t.strip())
    if tier_filter:
        tiers = [tier for tier in tiers if tier.tier_id in tier_filter]

    if not tiers:
        raise SystemExit("No tiers selected for local run.")

    output_root = Path(os.environ.get("OUTPUT_ROOT", repo_root / "outputs" / "local_multilingual_ablation"))
    output_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.setdefault("WANDB_MODE", "disabled")
    if args.device == "cpu":
        base_env["CUDA_VISIBLE_DEVICES"] = ""
        base_env["BF16"] = "False"
        base_env["FP16"] = "False"
        base_env["FLASH_ATTN"] = "disabled"

    for tier in tiers:
        tier_output = output_root / tier.tier_id
        tier_output.mkdir(parents=True, exist_ok=True)
        tier_num_experts = _count_experts_from_language_map(tier.language_map)

        if "lora" in allowed_kinds:
            for variant in lora_variants:
                output_dir = tier_output / f"lora_{variant.label}"
                env = base_env.copy()
                env.update(
                    {
                        "REPO_ROOT": str(repo_root),
                        "OUTPUT_DIR": str(output_dir),
                        "MODEL_NAME_OR_PATH": tier.model_path,
                        "TOKENIZED_PATH": tier.tokenized_path,
                        "MODEL_VARIANT": variant.label,
                        "LANGUAGE_TIER": tier.tier_id,
                    }
                )
                if variant.train_bs is not None:
                    env["PER_DEVICE_TRAIN_BATCH_SIZE"] = str(variant.train_bs)
                    env["PER_DEVICE_EVAL_BATCH_SIZE"] = str(variant.train_bs)
                _run_job(repo_root / "scripts" / "comparison" / "lora_job.sh", env, args.dry_run)

        if "hydra" in allowed_kinds:
            for variant in hydra_variants:
                output_dir = tier_output / f"hydra_{variant.label}"
                env = base_env.copy()
                env.update(
                    {
                        "REPO_ROOT": str(repo_root),
                        "OUTPUT_DIR": str(output_dir),
                        "MODEL_NAME_OR_PATH": tier.model_path,
                        "TOKENIZED_PATH": tier.tokenized_path,
                        "LANGUAGE_MAP": tier.language_map,
                        "MODEL_VARIANT": variant.label,
                        "LANGUAGE_TIER": tier.tier_id,
                        "USE_HYDRALORA_EXPERTS": str(variant.use_experts),
                        "HYDRALORA_NUM_EXPERTS": str(tier_num_experts if variant.use_experts else 1),
                        "HYDRALORA_TOP_K": str(variant.top_k),
                        "LORA_NUM": str(variant.lora_num),
                        "LANGUAGE_ROUTER_MODE": variant.router_mode,
                        "LANGUAGE_HEAD_ROUTER_MODE": variant.head_router_mode,
                        "LANGUAGE_PRIOR_WEIGHT": str(variant.prior_weight),
                        "LANGUAGE_BIAS_VALUE": str(variant.bias_value),
                        "LANGUAGE_HEAD_BIAS_VALUE": str(variant.head_bias_value),
                        "LANGUAGE_GUIDANCE_SCOPE": variant.guidance_scope,
                    }
                )
                if variant.train_bs is not None:
                    env["PER_DEVICE_TRAIN_BATCH_SIZE"] = str(variant.train_bs)
                    env["PER_DEVICE_EVAL_BATCH_SIZE"] = str(variant.train_bs)
                _run_job(repo_root / "scripts" / "comparison" / "hydralora_lpr_job.sh", env, args.dry_run)

        if "cola" in allowed_kinds:
            for variant in cola_variants:
                output_dir = tier_output / f"cola_{variant.label}"
                env = base_env.copy()
                env.update(
                    {
                        "REPO_ROOT": str(repo_root),
                        "OUTPUT_DIR": str(output_dir),
                        "MODEL_NAME_OR_PATH": tier.model_path,
                        "TOKENIZED_PATH": tier.tokenized_path,
                        "LANGUAGE_MAP": tier.language_map,
                        "MODEL_VARIANT": variant.label,
                        "LANGUAGE_TIER": tier.tier_id,
                        "USE_COLA_EXPERTS": str(variant.use_experts),
                        "COLA_NUM_EXPERTS": str(tier_num_experts if variant.use_experts else 1),
                        "COLA_NUM_A": str(variant.num_A),
                        "COLA_NUM_B": str(variant.num_B),
                        "COLA_STRATEGY": variant.strategy,
                        "COLA_TOP_K": str(variant.top_k),
                        "LANGUAGE_ROUTER_MODE": variant.router_mode,
                        "LANGUAGE_HEAD_ROUTER_MODE": variant.head_router_mode,
                        "LANGUAGE_PRIOR_WEIGHT": str(variant.prior_weight),
                        "LANGUAGE_BIAS_VALUE": str(variant.bias_value),
                        "LANGUAGE_HEAD_BIAS_VALUE": str(variant.head_bias_value),
                        "LANGUAGE_GUIDANCE_SCOPE": variant.guidance_scope,
                    }
                )
                if variant.train_bs is not None:
                    env["PER_DEVICE_TRAIN_BATCH_SIZE"] = str(variant.train_bs)
                    env["PER_DEVICE_EVAL_BATCH_SIZE"] = str(variant.train_bs)
                _run_job(repo_root / "scripts" / "comparison" / "cola_lpr_job.sh", env, args.dry_run)


if __name__ == "__main__":
    main()
