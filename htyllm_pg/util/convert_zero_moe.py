#!/usr/bin/env python3
"""
convert_zero_moe.py

Heuristic merger for DeepSpeed (ZeRO) shards, including many MoE-style shard files.
Produces a single pytorch state_dict (pytorch_model.bin) at the output path.

Usage:
    python htyllm_pg/util/convert_zero_moe.py \
        --checkpoint_dir ./checkpoints/step_4100 \
        --output_file ./hf_model/pytorch_model.bin \
        --verbose
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

def find_model_state_files(checkpoint_dir):
    p = Path(checkpoint_dir)
    if not p.exists():
        raise FileNotFoundError(checkpoint_dir)
    files = [str(x) for x in p.iterdir() if x.is_file() and 'model_state' in x.name or 'model_states' in x.name]
    # also include mp_rank and other patterns
    files += [str(x) for x in p.iterdir() if x.is_file() and re.search(r'mp_rank_\d+_model_states|_model_states\.pt|_model_state', x.name)]
    # dedupe
    files = sorted(set(files))
    return files

def extract_state_dict(container):
    """
    Try to find a dict of tensors inside container returned by torch.load.
    Returns a dict-like mapping param_name -> tensor
    """
    # Common keys DeepSpeed uses: 'module', 'model', 'state_dict', 'mp_rank_X_model_states' might be dict already
    if isinstance(container, dict):
        # if it *is* a mapping of tensors (leaf keys map to tensors), return as-is
        # Heuristic: if values are tensors or have 'dtype' attribute, treat as state_dict
        if all((torch.is_tensor(v) or isinstance(v, (int, float, str, bool, list, tuple)) or hasattr(v, 'dtype')) for v in container.values()):
            return container
        for candidate in ('module', 'model', 'state_dict', 'state_dict_mp', 'state'):
            if candidate in container and isinstance(container[candidate], dict):
                return container[candidate]
        # Some DeepSpeed checkpoints nest under 'sd' or 'optimizer_states' etc:
        for v in container.values():
            if isinstance(v, dict) and any(torch.is_tensor(x) for x in v.values()):
                # choose this one
                return v
    # fallback
    return {}

def try_concat(tensors):
    """
    Given a list of tensors (cpu), try to merge them:
    - If only one element -> return it
    - If multiple -> try concat dim 0, then dim 1, otherwise if all shapes equal return the first
    """
    if len(tensors) == 1:
        return tensors[0]
    # all tensors must be torch.Tensor
    tensors = [t.cpu() if torch.is_tensor(t) else torch.tensor(t) for t in tensors]
    shapes = [tuple(t.shape) for t in tensors]
    # try concat dim 0 if dims match after first
    try:
        if all(len(s) == len(shapes[0]) for s in shapes):
            # check if shapes match except in dim 0
            if all(s[1:] == shapes[0][1:] for s in shapes):
                return torch.cat(tensors, dim=0)
            # try dim 1
            if all(shapes[0][0] == s[0] for s in shapes) and all(s[2:] == shapes[0][2:] for s in shapes):
                return torch.cat(tensors, dim=1)
    except Exception:
        pass
    # if identical shapes, return first (could be replicated shards)
    if all(s == shapes[0] for s in shapes):
        return tensors[0]
    # last resort: stack
    try:
        return torch.cat(tensors, dim=0)
    except Exception:
        return tensors[0]

def merge_checkpoint(checkpoint_dir, output_file, verbose=False):
    files = find_model_state_files(checkpoint_dir)
    if verbose:
        print("Found files:", files)
    if not files:
        raise ValueError("No model state files found in checkpoint_dir")

    shards = defaultdict(list)  # param_name -> [tensor, ...]
    raw_keys_seen = set()
    for f in files:
        if verbose:
            print("Loading:", f)
        try:
            loaded = torch.load(f, map_location='cpu')
        except Exception as e:
            print(f"Warning: failed to torch.load {f}: {e}", file=sys.stderr)
            continue
        sd = extract_state_dict(loaded)
        if not sd:
            # Maybe the top-level is the state dict already (e.g. mp_rank_00_model_states.pt)
            if isinstance(loaded, dict):
                # try flatten
                sd = {k: v for k, v in loaded.items() if torch.is_tensor(v) or isinstance(v, dict)}
            else:
                sd = {}
        if not sd:
            if verbose:
                print(f"  No state-dict like object found inside {f}")
            continue
        # Flatten nested dict entries that are themselves containers holding tensors
        for k, v in sd.items():
            if isinstance(v, dict) and any(torch.is_tensor(x) for x in v.values()):
                # sometimes param stored under nested dict like {'param_name': {'fp32': tensor, 'fp16': tensor}}
                # try to choose the tensor value
                tensor_candidates = [x for x in v.values() if torch.is_tensor(x)]
                if tensor_candidates:
                    t = tensor_candidates[0]
                    shards[k].append(t)
                    raw_keys_seen.add(k)
                    if verbose:
                        print(f"  extracted nested tensor for key {k} from {f} shape={tuple(t.shape)}")
                    continue
            if torch.is_tensor(v):
                shards[k].append(v)
                raw_keys_seen.add(k)
                if verbose:
                    print(f"  got {k} shape={tuple(v.shape)}")
            # else ignore scalars/optimizer states
    if not shards:
        raise ValueError("No parameter tensors found in the checkpoint files")

    merged = {}
    problematic = []
    for k, list_of_tensors in shards.items():
        try:
            merged[k] = try_concat(list_of_tensors)
        except Exception as e:
            problematic.append((k, str(e)))
            merged[k] = list_of_tensors[0]
    # Save
    outp = Path(output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(outp))
    print(f"Saved merged state dict with {len(merged)} keys to {outp}")
    if problematic:
        print("Some keys had concat problems (kept first shard). Examples:")
        for k, e in problematic[:10]:
            print(" ", k, "->", e)
    # Print a small inventory
    print("Inventory (first 40 keys):")
    for i, k in enumerate(sorted(merged.keys())):
        print(f"  {i+1:3d}. {k} {tuple(merged[k].shape) if torch.is_tensor(merged[k]) else type(merged[k])}")
        if i >= 39:
            break

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    merge_checkpoint(args.checkpoint_dir, args.output_file, args.verbose)

if __name__ == "__main__":
    main()
