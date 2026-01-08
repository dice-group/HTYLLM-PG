import json
import os
import sys
import inspect
import torch
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist

RUNS = {
    "cola_local_lpr": {
        "root": "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-lpr_20260108_031230",
        "checkpoints": ["checkpoint-40_adapter_sharded", "checkpoint-80_adapter_sharded"],
    },
}

CONFIG_KEYS = [
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "fan_in_fan_out",
    "modules_to_save",
    "use_rslora",
    "rank_pattern",
    "alpha_pattern",
    "lora_type",
    "lora_expert_num",
    "language_list",
    "language_router_mode",
    "language_guidance_scope",
    "language_prior_weight",
    "cola_num_experts",
    "cola_num_a",
    "cola_num_b",
    "cola_strategy",
    "hydralora_num_experts",
    "lora_num",
]


def load_config(config_path):
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_config(config):
    summary = {}
    for key in CONFIG_KEYS:
        if key in config:
            summary[key] = config[key]
    return summary


def ensure_dist():
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def load_full_state(ckpt_dir):
    ensure_dist()
    checkpoint_id = None
    meta_path = os.path.join(ckpt_dir, "adapter_sharded.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        checkpoint_id = meta.get("checkpoint_id")
    try:
        from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    except Exception as exc:  # pragma: no cover - depends on torch version
        raise RuntimeError(
            "torch.distributed.checkpoint._EmptyStateDictLoadPlanner is required "
            "to load a full state dict without a model."
        ) from exc
    planner = _EmptyStateDictLoadPlanner()
    state = {}
    if hasattr(dist_cp, "load_state_dict"):
        load_fn = dist_cp.load_state_dict
        params = inspect.signature(load_fn).parameters
        if not hasattr(dist_cp, "FileSystemReader"):
            raise RuntimeError("torch.distributed.checkpoint.FileSystemReader not available")
        reader = dist_cp.FileSystemReader(ckpt_dir)
        kwargs = {}
        if "planner" in params:
            kwargs["planner"] = planner
        load_fn(state, reader, **kwargs)
        return state
    if hasattr(dist_cp, "load"):
        load_fn = dist_cp.load
        params = inspect.signature(load_fn).parameters
        kwargs = {}
        if "planner" in params:
            kwargs["planner"] = planner
        if "storage_reader" in params and hasattr(dist_cp, "FileSystemReader"):
            reader = dist_cp.FileSystemReader(ckpt_dir)
            kwargs["storage_reader"] = reader
            if checkpoint_id is not None and "checkpoint_id" in params:
                kwargs["checkpoint_id"] = checkpoint_id
        else:
            if "checkpoint_id" in params:
                kwargs["checkpoint_id"] = checkpoint_id or ckpt_dir
        load_fn(state, **kwargs)
        return state
    raise RuntimeError("torch.distributed.checkpoint load API not available")


def flatten_tensors(obj, prefix=""):
    tensors = {}
    if torch.is_tensor(obj):
        tensors[prefix or "tensor"] = obj
        return tensors
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            tensors.update(flatten_tensors(value, next_prefix))
        return tensors
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            tensors.update(flatten_tensors(value, next_prefix))
        return tensors
    return tensors


def load_sharded_state(ckpt_dir):
    state = load_full_state(ckpt_dir)
    return flatten_tensors(state)


def fingerprint_tensors(tensors):
    if not tensors:
        return {"tensor_keys": 0, "norm_sum": 0.0, "shape_sum": 0}
    norm_sum = 0.0
    shape_sum = 0
    for tensor in tensors.values():
        norm_sum += float(tensor.float().norm().item())
        shape_sum += tensor.numel()
    return {
        "tensor_keys": len(tensors),
        "norm_sum": norm_sum,
        "shape_sum": shape_sum,
    }


def compare_two(checkpoint_a, checkpoint_b):
    tensors_a = load_sharded_state(checkpoint_a)
    tensors_b = load_sharded_state(checkpoint_b)
    fp_a = fingerprint_tensors(tensors_a)
    fp_b = fingerprint_tensors(tensors_b)

    shared_keys = set(tensors_a.keys()) & set(tensors_b.keys())
    diff_norm = 0.0
    for key in sorted(shared_keys):
        diff_norm += float((tensors_a[key].float() - tensors_b[key].float()).norm().item())
    return tensors_a, tensors_b, fp_a, fp_b, diff_norm


def collect_keys(tensors, patterns):
    return {k for k in tensors.keys() if any(p in k for p in patterns)}


def summarize_group_diffs(tensors_a, tensors_b, label, patterns):
    keys_a = collect_keys(tensors_a, patterns)
    keys_b = collect_keys(tensors_b, patterns)
    if not keys_a and not keys_b:
        print(f"{label}: no matching keys for patterns={patterns}")
        return
    shared = keys_a & keys_b
    missing = keys_a - keys_b
    extra = keys_b - keys_a
    diff = 0.0
    for key in shared:
        diff += float((tensors_a[key].float() - tensors_b[key].float()).norm().item())
    sample_keys = ", ".join(sorted(shared)[:3])
    key_match = not missing and not extra
    print(
        f"{label}: key_match={key_match} shared={len(shared)} "
        f"A_only={len(missing)} B_only={len(extra)} sum_l2_diff={diff:.6f} sample={sample_keys}"
    )
    if not key_match:
        missing_sample = ", ".join(sorted(missing)[:3])
        extra_sample = ", ".join(sorted(extra)[:3])
        print(f"{label} key_mismatch sample missing={missing_sample} extra={extra_sample}")


def print_key_hints(tensors, label):
    substrings = ["lora", "cola", "router", "gate", "expert", "adapter"]
    hints = [k for k in tensors.keys() if any(s in k.lower() for s in substrings)]
    print(f"{label}: key_hints={len(hints)} sample={', '.join(sorted(hints)[:10])}")


def main():
    for name, spec in RUNS.items():
        root = spec["root"]
        ckpt_a, ckpt_b = spec["checkpoints"]
        ckpt_a_path = os.path.join(root, ckpt_a)
        ckpt_b_path = os.path.join(root, ckpt_b)

        if not os.path.isdir(ckpt_a_path) or not os.path.isdir(ckpt_b_path):
            print(f"[SKIP] {name}: missing checkpoints {ckpt_a_path} or {ckpt_b_path}")
            continue

        config_a = summarize_config(load_config(os.path.join(ckpt_a_path, "adapter_config.json")))
        config_b = summarize_config(load_config(os.path.join(ckpt_b_path, "adapter_config.json")))
        config_match = config_a == config_b

        print(f"\n=== {name} ===")
        print(f"checkpoint A: {ckpt_a_path}")
        print(f"checkpoint B: {ckpt_b_path}")
        print(f"config_match: {config_match}")
        if not config_match:
            print("config_a:", config_a)
            print("config_b:", config_b)
        if "cola_num_a" in config_a and "cola_num_b" in config_a:
            cola_ratio_ok = config_a["cola_num_b"] == 3 * config_a["cola_num_a"]
            print(
                f"cola A/B ratio: A={config_a['cola_num_a']} B={config_a['cola_num_b']} "
                f"expected_B=3*A -> {cola_ratio_ok}"
            )

        tensors_a, tensors_b, fp_a, fp_b, diff_norm = compare_two(ckpt_a_path, ckpt_b_path)
        print(f"A tensors: {fp_a}")
        print(f"B tensors: {fp_b}")
        print(f"sum L2 diff across shared tensors: {diff_norm:.6f}")
        update_changed = diff_norm > 1e-6
        print(f"update_check: changed={update_changed}")
        keys_a = set(tensors_a.keys())
        keys_b = set(tensors_b.keys())
        key_set_match = keys_a == keys_b
        print(f"key_set_match: {key_set_match} total_A={len(keys_a)} total_B={len(keys_b)}")
        if not key_set_match:
            missing = ", ".join(sorted(keys_a - keys_b)[:3])
            extra = ", ".join(sorted(keys_b - keys_a)[:3])
            print(f"key_set_mismatch sample missing={missing} extra={extra}")

        print_key_hints(tensors_a, "A key hints")
        print_key_hints(tensors_b, "B key hints")

        summarize_group_diffs(
            tensors_a,
            tensors_b,
            "LoRA A diff",
            ["lora_A", "lora_a", "cola_A", "cola_a"],
        )
        summarize_group_diffs(
            tensors_a,
            tensors_b,
            "LoRA B diff",
            ["lora_B", "lora_b", "cola_B", "cola_b"],
        )
        summarize_group_diffs(
            tensors_a,
            tensors_b,
            "Router diff",
            ["router", "gate", "language_router", "routing"],
        )

        a_keys_a = collect_keys(tensors_a, ["lora_A", "lora_a", "cola_A", "cola_a"])
        b_keys_a = collect_keys(tensors_a, ["lora_B", "lora_b", "cola_B", "cola_b"])
        a_keys_b = collect_keys(tensors_b, ["lora_A", "lora_a", "cola_A", "cola_a"])
        b_keys_b = collect_keys(tensors_b, ["lora_B", "lora_b", "cola_B", "cola_b"])
        if a_keys_a and b_keys_a and a_keys_b and b_keys_b:
            ratio_a = len(b_keys_a) / max(len(a_keys_a), 1)
            ratio_b = len(b_keys_b) / max(len(a_keys_b), 1)
            ratio_match = abs(ratio_a - ratio_b) < 1e-6
            print(
                f"A/B key count: "
                f"A={len(a_keys_a)} B={len(b_keys_a)} "
                f"ratio_A={ratio_a:.2f} ratio_B={ratio_b:.2f} ratio_match={ratio_match}"
            )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
