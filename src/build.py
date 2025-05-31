

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
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


def ffn_params_per_expert(cfg: MixtralConfig) -> int:
    """Calculate parameters per expert in MoE layers.
    
    Each expert has 3 linear layers (typically with bias=False).
    """
    return 3 * cfg.hidden_size * cfg.intermediate_size


def count_active_params(total_params: int, cfg: MixtralConfig) -> int:
    """Calculate active parameters during inference (with only top-k experts)."""
    ffn_per_exp = ffn_params_per_expert(cfg)
    num_layers = cfg.num_hidden_layers
    
    # All FFN weights stored on disk
    total_ffn_params = ffn_per_exp * cfg.num_local_experts * num_layers
    
    # Keep only top-k experts per layer at run time
    active_ffn_params = ffn_per_exp * cfg.num_experts_per_tok * num_layers
    
    shared_params = total_params - total_ffn_params
    return shared_params + active_ffn_params


def count_parameters(model: torch.nn.Module, cfg: MixtralConfig) -> Dict[str, Any]:
    """Count total, trainable, and active parameters for a Mixtral model.
    
    Returns:
        Dictionary with parameter counts (total, trainable, active).
    """
    # 1) Total parameters (includes frozen weights, experts, everything)
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2) Trainable parameters only
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 3) Active parameters per token (MoE-specific)
    active_params = count_active_params(total_params, cfg)
    
    return {
        "total": total_params,
        "total_millions": total_params / 1e6,
        "trainable": trainable_params,
        "trainable_millions": trainable_params / 1e6,
        "active": active_params,
        "active_millions": active_params / 1e6,
    }


def print_parameter_counts(model: torch.nn.Module, cfg: MixtralConfig) -> None:
    """Print total, trainable, and active parameters for a Mixtral model."""
    param_counts = count_parameters(model, cfg)
    
    print(f"Total params: {param_counts['total']:,}  ({param_counts['total_millions']:.2f} M)")
    print(f"Trainable params: {param_counts['trainable']:,}  ({param_counts['trainable_millions']:.2f} M)")
    print(f"~Active params/token: {param_counts['active']:,}  ({param_counts['active_millions']:.2f} M)")


def cli():
    import argparse
    ap = argparse.ArgumentParser("init tiny mixtral model")
    ap.add_argument("--config", default=None, help="Path to JSON config (optional)")
    ap.add_argument("--save_dir", default="checkpoints/init")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = MixtralConfig.from_json_file(args.config) if args.config else tiny_mixtral_config()
    model = build_model(cfg, args.device)
    
    # Print parameter counts
    print("Parameter counts:")
    print_parameter_counts(model, cfg)
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_dir)
    print("Model saved to", Path(args.save_dir).resolve())


if __name__ == "__main__":
    cli()