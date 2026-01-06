#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch


def _parse_torch_dtype(value: Optional[str]):
    if value is None:
        return None
    val = str(value).strip().lower()
    if val in ("", "none", "null"):
        return None
    if val == "auto":
        return "auto"
    if val in ("bf16", "bfloat16"):
        return torch.bfloat16
    if val in ("fp16", "float16", "half"):
        return torch.float16
    if val in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def _ensure_repo_peft():
    repo_root = Path(__file__).resolve().parents[1]
    peft_root = repo_root / "LLaMA-Factory" / "src"
    if str(peft_root) not in sys.path:
        sys.path.insert(0, str(peft_root))


def _init_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)


def main():
    parser = argparse.ArgumentParser(description="Merge sharded adapter checkpoint saved by DCP.")
    parser.add_argument("--adapter-sharded-dir", required=True, help="Path to *_adapter_sharded directory")
    parser.add_argument("--output-dir", default=None, help="Output adapter dir (default: *_adapter)")
    parser.add_argument("--base-model", default=None, help="Override base model name/path")
    parser.add_argument("--torch-dtype", default=None, help="bf16/fp16/fp32/auto")
    parser.add_argument("--device-map", default=None, help="Transformers device_map (single-process only)")
    parser.add_argument("--no-safetensors", action="store_true", help="Save .bin instead of safetensors")
    args = parser.parse_args()

    adapter_sharded_dir = Path(args.adapter_sharded_dir).resolve()
    if not adapter_sharded_dir.exists():
        raise FileNotFoundError(f"Adapter sharded dir not found: {adapter_sharded_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(str(adapter_sharded_dir).replace("_adapter_sharded", "_adapter"))
    output_dir = output_dir.resolve()

    _ensure_repo_peft()
    _init_distributed()

    rank = 0
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    if args.device_map and world_size > 1:
        raise ValueError("--device-map is only supported for single-process merges")

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from torch.distributed.checkpoint import FileSystemReader, load as dcp_load
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
    from transformers import AutoModelForCausalLM

    peft_config = PeftConfig.from_pretrained(str(adapter_sharded_dir))
    base_model = args.base_model or peft_config.base_model_name_or_path
    if not base_model:
        raise ValueError("Base model path is required (missing in adapter config and --base-model)")

    load_kwargs = {"low_cpu_mem_usage": True}
    torch_dtype = _parse_torch_dtype(args.torch_dtype)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if not args.device_map:
        base.to(device)
    peft_model = get_peft_model(base, peft_config)

    if world_size > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        peft_model = FSDP(peft_model, use_orig_params=True)

    load_opts = StateDictOptions(full_state_dict=False, cpu_offload=False, ignore_frozen_params=True)
    shard_state = get_model_state_dict(peft_model, options=load_opts)
    reader = FileSystemReader(str(adapter_sharded_dir))
    if world_size > 1:
        dcp_load(shard_state, storage_reader=reader)
    else:
        dcp_load(shard_state, storage_reader=reader, no_dist=True)
    set_model_state_dict(peft_model, shard_state, options=load_opts)

    gather_opts = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
    full_state = get_model_state_dict(peft_model, options=gather_opts)
    if full_state is not None and any(k.startswith("_fsdp_wrapped_module.") for k in full_state):
        full_state = {k.removeprefix("_fsdp_wrapped_module."): v for k, v in full_state.items()}

    if world_size > 1:
        unwrapped = peft_model.module
    else:
        unwrapped = peft_model

    adapter_state = get_peft_model_state_dict(unwrapped, state_dict=full_state)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(
            str(output_dir),
            state_dict=adapter_state,
            safe_serialization=not args.no_safetensors,
        )
        meta_path = output_dir / "adapter_merge.json"
        meta_path.write_text(
            json.dumps(
                {
                    "source": str(adapter_sharded_dir),
                    "base_model_name_or_path": base_model,
                    "world_size": world_size,
                },
                indent=2,
            )
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
