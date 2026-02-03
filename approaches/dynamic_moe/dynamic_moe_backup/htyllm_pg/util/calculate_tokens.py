#!/usr/bin/env python3
"""Calculate the number of tokens processed during training."""

import argparse
import json
from pathlib import Path


def format_tokens(tokens: int) -> str:
    """Format token count in human-readable form."""
    if tokens >= 1e12:
        return f"{tokens / 1e12:.2f}T"
    elif tokens >= 1e9:
        return f"{tokens / 1e9:.2f}B"
    elif tokens >= 1e6:
        return f"{tokens / 1e6:.2f}M"
    elif tokens >= 1e3:
        return f"{tokens / 1e3:.2f}K"
    return str(tokens)


def load_ds_config(config_path: Path) -> dict:
    """Load DeepSpeed config if it exists."""
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Calculate tokens processed during training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m htyllm_pg.util.calculate_tokens --steps 50000
  python -m htyllm_pg.util.calculate_tokens --steps 50000 --gpus 16 --seq-len 2048
        """
    )
    parser.add_argument("--steps", type=int, required=True,
                        help="Number of training steps completed")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="Micro batch size per GPU (default: from ds_config.json or 6)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (default: from ds_config.json or 16)")
    parser.add_argument("--gpus", type=int, default=16,
                        help="Total number of GPUs (default: 16)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--ds-config", type=str, default="ds_config.json",
                        help="Path to DeepSpeed config (default: ds_config.json)")
    
    args = parser.parse_args()
    
    # Try to load defaults from ds_config.json
    ds_config = load_ds_config(Path(args.ds_config))
    
    micro_batch = args.micro_batch
    if micro_batch is None:
        micro_batch = ds_config.get("train_micro_batch_size_per_gpu", 6)
    
    grad_accum = args.grad_accum
    if grad_accum is None:
        grad_accum = ds_config.get("gradient_accumulation_steps", 16)
    
    # Effective sequence length (input_ids = seq[:-1] in dataset)
    effective_seq_len = args.seq_len - 1
    
    # Calculate tokens
    tokens_per_step = micro_batch * grad_accum * args.gpus * effective_seq_len
    total_tokens = args.steps * tokens_per_step
    
    # Print results
    print("\nTraining Configuration:")
    print(f"  - Steps: {args.steps:,}")
    print(f"  - Micro batch per GPU: {micro_batch}")
    print(f"  - Gradient accumulation: {grad_accum}")
    print(f"  - GPUs: {args.gpus}")
    print(f"  - Sequence length: {effective_seq_len} (effective)")
    print(f"\nTokens per step: {tokens_per_step:,}")
    print(f"Total tokens processed: {total_tokens:,} (~{format_tokens(total_tokens)} tokens)\n")


if __name__ == "__main__":
    main()
