from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from llamafactory.extras.language import build_language_vocab, language_to_ids, load_language_groupings
from scripts.comparison.ablation_specs import (
    default_ablation_script_path,
    parse_cola_variants,
    parse_hydra_variants,
    parse_lora_variants,
)


RUN_TRAIN_SMOKE = os.environ.get("RUN_TRAIN_SMOKE", "") == "1"


def _build_tokenized_dataset(
    tmp_path: Path,
    model_name_or_path: str,
    language_map_path: Path,
    repo_root: Path,
) -> Path:
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")

    tokenized_root = tmp_path / "tokenized"
    tokenized_root.mkdir(parents=True, exist_ok=True)

    language_map, _, _, _ = load_language_groupings(str(language_map_path))
    if not language_map:
        raise ValueError(f"Failed to load language map from {language_map_path}")
    language_vocab, family_vocab = build_language_vocab(language_map)

    dataset_name = os.environ.get("SMOKE_DATASET_NAME", "c4")
    dataset_config = os.environ.get("SMOKE_DATASET_CONFIG", "multilingual")
    dataset_split = os.environ.get("SMOKE_DATASET_SPLIT", "train")
    max_samples = int(os.environ.get("SMOKE_DATASET_MAX_SAMPLES", "32"))
    max_length = int(os.environ.get("SMOKE_DATASET_MAX_LENGTH", "128"))
    max_scan = int(os.environ.get("SMOKE_DATASET_MAX_SCAN", "1000"))
    data_files = os.environ.get("SMOKE_DATASET_DATA_FILES")
    fallback_language = os.environ.get("SMOKE_DATASET_DEFAULT_LANGUAGE")
    allow_fallback = os.environ.get("SMOKE_ALLOW_FALLBACK_LANGUAGE", "1") == "1"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    stream = None
    if data_files:
        stream = datasets.load_dataset(
            "json",
            data_files=data_files.split(","),
            split="train",
            streaming=True,
        )
    else:
        try:
            stream = datasets.load_dataset(
                dataset_name,
                dataset_config,
                split=dataset_split,
                streaming=True,
                trust_remote_code=True,
            )
        except RuntimeError as exc:
            if "Dataset scripts are no longer supported" not in str(exc):
                raise
            local_fallback = repo_root / "LLaMA-Factory" / "data" / "c4_demo.jsonl"
            if not local_fallback.exists():
                raise
            stream = datasets.load_dataset(
                "json",
                data_files=[str(local_fallback)],
                split="train",
                streaming=True,
            )

    if stream is None:
        raise ValueError("Failed to build dataset stream.")

    if not fallback_language:
        fallback_language = next(iter(language_vocab))

    input_ids = []
    attention_masks = []
    labels = []
    language_ids = []
    family_ids = []
    seen = 0

    for example in stream:
        seen += 1
        lang = example.get("lang") or example.get("language")
        if lang is None and allow_fallback:
            lang = fallback_language
        if lang is None:
            if seen >= max_scan:
                break
            continue
        lang_id, fam_id = language_to_ids(lang, language_map, language_vocab, family_vocab)
        if (lang_id < 0 or fam_id < 0) and allow_fallback:
            lang_id, fam_id = language_to_ids(
                fallback_language, language_map, language_vocab, family_vocab
            )
        if lang_id < 0 or fam_id < 0:
            if seen >= max_scan:
                break
            continue
        text = example.get("text")
        if not text:
            if seen >= max_scan:
                break
            continue
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        if not encoded.get("input_ids"):
            if seen >= max_scan:
                break
            continue
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        labels.append(encoded["input_ids"])
        language_ids.append(lang_id)
        family_ids.append(fam_id)
        if len(input_ids) >= max_samples:
            break
        if seen >= max_scan:
            break

    if not input_ids:
        raise ValueError(
            "No usable examples found in dataset stream; check dataset access and language map coverage."
        )

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "language_ids": language_ids,
        "family_ids": family_ids,
    }

    train = datasets.Dataset.from_dict(data)
    valid = datasets.Dataset.from_dict(data)
    dataset = datasets.DatasetDict(train=train, validation=valid)
    dataset.save_to_disk(tokenized_root)
    return tokenized_root


def _run_train(
    *,
    repo_root: Path,
    output_dir: Path,
    tokenized_path: Path,
    model_name_or_path: str,
    finetuning_type: str,
    extra_args: list[str],
    env: dict[str, str],
) -> None:
    cmd = [
        "python3",
        "-m",
        "llamafactory.cli",
        "train",
        "--stage",
        "sft",
        "--do_train",
        "--model_name_or_path",
        model_name_or_path,
        "--dataset",
        "tokenized_smoke",
        "--dataset_dir",
        str(repo_root / "LLaMA-Factory" / "data"),
        "--template",
        "llama3",
        "--finetuning_type",
        finetuning_type,
        "--output_dir",
        str(output_dir),
        "--overwrite_output_dir",
        "--max_steps",
        "2",
        "--logging_steps",
        "1",
        "--eval_strategy",
        "no",
        "--save_steps",
        "2",
        "--save_safetensors",
        "False",
        "--save_only_model",
        "True",
        "--seed",
        "42",
        "--tokenized_path",
        str(tokenized_path),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "2e-4",
        "--lora_rank",
        "4",
        "--lora_alpha",
        "8",
        "--lora_dropout",
        "0.0",
        "--lora_target",
        "q_proj,k_proj,v_proj,o_proj",
        "--bf16",
        "False",
        "--fp16",
        "False",
        "--flash_attn",
        "disabled",
        "--report_to",
        "none",
        "--include_effective_tokens_per_second",
        "true",
        "--include_num_input_tokens_seen",
        "true",
    ]
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True, env=env)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_training_outputs(output_dir: Path) -> dict:
    trainer_state = output_dir / "trainer_state.json"
    train_results = output_dir / "train_results.json"
    assert trainer_state.exists(), f"Missing trainer_state.json in {output_dir}"
    assert train_results.exists(), f"Missing train_results.json in {output_dir}"
    state = _load_json(trainer_state)
    results = _load_json(train_results)
    assert results.get("train_loss") is not None
    assert results.get("train_runtime", 0) > 0
    assert results.get("effective_tokens_per_sec", 0) > 0
    assert state.get("global_step", 0) >= 1
    return state


def _assert_metrics_present(state: dict, prefixes: tuple[str, ...], keys: tuple[str, ...]) -> None:
    history = state.get("log_history", [])
    found = {key: False for key in keys}
    found_prefix = {prefix: False for prefix in prefixes}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for key in keys:
            if key in entry:
                found[key] = True
        for prefix in prefixes:
            if any(k.startswith(prefix) for k in entry.keys()):
                found_prefix[prefix] = True
    missing = [k for k, ok in found.items() if not ok]
    assert not missing, f"Missing metrics in log_history: {missing}"
    if prefixes and not any(found_prefix.values()):
        raise AssertionError(f"Missing metrics with any prefix {prefixes}")


@pytest.mark.skipif(not RUN_TRAIN_SMOKE, reason="Set RUN_TRAIN_SMOKE=1 to enable training smoke tests.")
def test_smoke_training_all_variants(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env["PYTHONPATH"] = f"{repo_root / 'LLaMA-Factory' / 'src'}:{env.get('PYTHONPATH', '')}"

    model_name_or_path = env.get("MODEL_NAME_OR_PATH", "meta-llama/Llama-3.2-1B")
    output_root = Path(env.get("SMOKE_OUTPUT_ROOT", tmp_path))
    output_root.mkdir(parents=True, exist_ok=True)
    language_map = repo_root / "tools" / "two_stage_clustering" / "12_tier_language_groupings.json"
    tokenized_path = _build_tokenized_dataset(output_root / "tokenized", model_name_or_path, language_map, repo_root)

    script_path = default_ablation_script_path()
    cola_variants = parse_cola_variants(script_path, include_commented=True)
    hydra_variants = parse_hydra_variants(script_path, include_commented=True)
    lora_variants = parse_lora_variants(script_path, include_commented=True)

    for variant in lora_variants:
        output_dir = output_root / f"lora_{variant.label}"
        _run_train(
            repo_root=repo_root,
            output_dir=output_dir,
            tokenized_path=tokenized_path,
            model_name_or_path=model_name_or_path,
            finetuning_type="lora",
            extra_args=[],
            env=env,
        )
        _assert_training_outputs(output_dir)

    for variant in cola_variants:
        output_dir = output_root / f"cola_{variant.label}"
        extra_args = [
            "--num_A",
            str(variant.num_A),
            "--num_B",
            str(variant.num_B),
            "--cola_strategy",
            variant.strategy,
            "--use_cola_experts",
            str(variant.use_experts),
            "--cola_num_experts",
            "4",
            "--cola_top_k",
            str(variant.top_k),
            "--language_column",
            "language",
            "--language_map",
            str(language_map),
            "--language_router_mode",
            variant.router_mode,
            "--language_head_router_mode",
            variant.head_router_mode,
            "--language_prior_weight",
            str(variant.prior_weight),
            "--language_bias_value",
            str(variant.bias_value),
            "--language_head_bias_value",
            str(variant.head_bias_value),
            "--language_guidance_scope",
            variant.guidance_scope,
        ]
        _run_train(
            repo_root=repo_root,
            output_dir=output_dir,
            tokenized_path=tokenized_path,
            model_name_or_path=model_name_or_path,
            finetuning_type="cola",
            extra_args=extra_args,
            env=env,
        )
        state = _assert_training_outputs(output_dir)
        if variant.prior_weight > 0:
            _assert_metrics_present(
                state,
                prefixes=("cola/", "train/cola/"),
                keys=("language_prior_loss",),
            )

    for variant in hydra_variants:
        output_dir = output_root / f"hydra_{variant.label}"
        extra_args = [
            "--lora_num",
            str(variant.lora_num),
            "--use_hydralora_experts",
            str(variant.use_experts),
            "--hydralora_num_experts",
            "4",
            "--hydralora_top_k",
            str(variant.top_k),
            "--language_column",
            "language",
            "--language_map",
            str(language_map),
            "--language_router_mode",
            variant.router_mode,
            "--language_head_router_mode",
            variant.head_router_mode,
            "--language_prior_weight",
            str(variant.prior_weight),
            "--language_bias_value",
            str(variant.bias_value),
            "--language_head_bias_value",
            str(variant.head_bias_value),
            "--language_guidance_scope",
            variant.guidance_scope,
        ]
        _run_train(
            repo_root=repo_root,
            output_dir=output_dir,
            tokenized_path=tokenized_path,
            model_name_or_path=model_name_or_path,
            finetuning_type="hydralora",
            extra_args=extra_args,
            env=env,
        )
        state = _assert_training_outputs(output_dir)
        if variant.prior_weight > 0:
            _assert_metrics_present(
                state,
                prefixes=("hydralora/", "train/hydralora/"),
                keys=("language_prior_loss",),
            )
