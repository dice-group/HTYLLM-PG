"""Helpers to create a downsized (7×17 M) Mixtral MoE model."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
from transformers import MixtralConfig, MixtralForCausalLM


def tiny_mixtral_config(
    vocab_size: int = 131_072,
    d_model: int = 384,
    d_ff: int = 1_536,
    n_layers: int = 12,
    n_heads: int = 6,
    n_kv: int = 2,
    n_experts: int = 7,
    top_k: int = 2,
    max_pos: int = 8_192,
    router_aux_coef: float = 1e-3,
):
    return MixtralConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=d_ff,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        num_local_experts=n_experts,
        num_experts_per_tok=top_k,
        router_aux_loss_coef=router_aux_coef,
        router_jitter_noise=0.1,
        max_position_embeddings=max_pos,
    )


def build_model(cfg: MixtralConfig, device: Optional[str | torch.device] = None):
    model = MixtralForCausalLM(cfg)
    if device:
        model.to(device)
    return model


def cli():
    import argparse
    ap = argparse.ArgumentParser("init tiny mixtral model")
    ap.add_argument("--config", default=None, help="Path to JSON config (optional)")
    ap.add_argument("--save_dir", default="checkpoints/init")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = MixtralConfig.from_json_file(args.config) if args.config else tiny_mixtral_config()
    model = build_model(cfg, args.device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_dir)
    print("Model saved to", Path(args.save_dir).resolve())


if __name__ == "__main__":
    cli()