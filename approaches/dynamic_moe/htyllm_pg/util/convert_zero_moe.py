#!/usr/bin/env python3
"""
convert_zero_moe.py

Merge DeepSpeed ZeRO+MoE checkpoint shards into a single PyTorch state_dict
matching the `moe_builder` naming from training hyperparameters.

Usage:
  python convert_zero_moe.py \
    --checkpoint_dir ./checkpoints/step_4100 \
    --output_file ./hf_model/pytorch_model.bin \
    --verbose
"""

import argparse
import os, re, sys
from pathlib import Path
from collections import defaultdict

import torch
from htyllm_pg.model_builder import moe_builder


def find_model_state_files(checkpoint_dir):
    p = Path(checkpoint_dir)
    files = [str(x) for x in p.iterdir() if x.is_file() and ('model_state' in x.name or re.search(r'mp_rank_\d+_model_states|_model_states\.pt|_model_state', x.name))]
    return sorted(set(files))


def extract_state_dict(container):
    """Return a dict of param_name -> tensor inside loaded checkpoint."""
    if isinstance(container, dict):
        # Heuristic: leaf dict contains tensors
        if all((torch.is_tensor(v) or hasattr(v, "dtype")) for v in container.values()):
            return container
        for candidate in ("module", "model", "state_dict", "state_dict_mp", "state"):
            if candidate in container and isinstance(container[candidate], dict):
                return container[candidate]
        # fallback: find first dict containing tensors
        for v in container.values():
            if isinstance(v, dict) and any(torch.is_tensor(x) for x in v.values()):
                return v
    return {}


def try_concat(tensors):
    """Concat shards along dim=0 or 1, or return first if cannot concat."""
    if len(tensors) == 1:
        return tensors[0]
    tensors = [t.cpu() for t in tensors]
    shapes = [t.shape for t in tensors]
    try:
        if all(len(s) == len(shapes[0]) for s in shapes):
            if all(s[1:] == shapes[0][1:] for s in shapes):
                return torch.cat(tensors, dim=0)
            if all(s[0] == shapes[0][0] for s in shapes) and all(s[2:] == shapes[0][2:] for s in shapes):
                return torch.cat(tensors, dim=1)
    except Exception:
        pass
    if all(s == shapes[0] for s in shapes):
        return tensors[0]
    try:
        return torch.cat(tensors, dim=0)
    except Exception:
        return tensors[0]


def merge_checkpoint(checkpoint_dir, verbose=False):
    files = find_model_state_files(checkpoint_dir)
    if verbose:
        print("Found checkpoint files:", files)
    shards = defaultdict(list)
    for f in files:
        if verbose:
            print("Loading:", f)
        loaded = torch.load(f, map_location="cpu")
        sd = extract_state_dict(loaded)
        if not sd:
            continue
        for k, v in sd.items():
            if torch.is_tensor(v):
                shards[k].append(v)
            elif isinstance(v, dict):
                tensor_candidates = [x for x in v.values() if torch.is_tensor(x)]
                if tensor_candidates:
                    shards[k].append(tensor_candidates[0])
    merged = {}
    for k, lst in shards.items():
        merged[k] = try_concat(lst)
    return merged


def remap_keys_for_model(merged_state_dict):
    """Heuristic remap DeepSpeed/MoE keys to `moe_builder` names."""
    new_sd = {}
    for k, v in merged_state_dict.items():
        new_k = k
        # remove `module.` prefix if exists
        new_k = new_k.replace("module.", "")
        # replace DeepSpeed style layers naming to moe_builder naming
        new_k = new_k.replace("layers.", "transformer.layers.")
        # optional: handle MoE expert prefixes if necessary
        new_sd[new_k] = v
    return new_sd


def build_model_from_hyperparams(args):
    model = moe_builder(
        vocab_size=262144,
        max_seq_len=2048,
        dim=2048,
        depth=24,
        heads=16,
        mlp_dim=8192,
        dim_head=128,
        dropout=0.0,
        emb_dropout=0.0,
        moe_layers=[3,7,11,15,19,23],
        num_experts=8,
        k=-1,
        capacity_factor=1.5,
        eval_capacity_factor=2.0,
        min_capacity=0.0,
        use_residual=False,
        gate_backward="ste",
        ep_size=1,
        topany_gating_impl="sparse",
        use_flash_attention=True,
        use_gradient_checkpointing=True
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--verbose", action="store_true")

    # Hyperparameters from your SLURM training config
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=1200)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=8192)
    parser.add_argument("--dim_head", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--emb_dropout", type=float, default=0.0)
    parser.add_argument("--moe_layers", nargs="+", type=int, default=[3,7,11,15,19,23])
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--k", type=int, default=-1)
    parser.add_argument("--capacity_factor", type=float, default=1.5)
    parser.add_argument("--eval_capacity_factor", type=float, default=2.0)
    parser.add_argument("--min_capacity", type=float, default=0.0)
    parser.add_argument("--use_residual", type=bool, default=False)
    parser.add_argument("--gate_backward", type=str, default="ste")
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--topany_gating_impl", type=str, default="sparse")
    parser.add_argument("--use_flash_attention", type=bool, default=True)
    args = parser.parse_args()

    merged = merge_checkpoint(args.checkpoint_dir, verbose=args.verbose)
    if args.verbose:
        print(f"Merged {len(merged)} parameters")

    mapped = remap_keys_for_model(merged)

    outp = Path(args.output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, str(outp))
    print(f"Saved merged + mapped state dict to {outp}")


if __name__ == "__main__":
    main()
