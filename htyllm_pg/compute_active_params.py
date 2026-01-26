#!/usr/bin/env python3
"""
Compute average active parameters for an MoE model.

For MoE models, not all parameters are active for every token.
This script loads a checkpoint, runs inference over the dataset,
and calculates average active parameters based on expert activation patterns.

Usage:
    python htyllm_pg/compute_active_params.py \
        --checkpoint-dir /scratch/hpc-prf-merlin/luke/checkpoints_multilingual_3_5b \
        --checkpoint-step 124000 \
        --data-dir /scratch/hpc-prf-merlin/luke/tokenized_multilingual \
        --dim 3072 --depth 28 --heads 24 --dim-head 128 --mlp-dim 12288 \
        --moe-layers 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
        --num-experts 8 --ep-size 8 --use-flash-attention --l1-lambda 0.0001 \
        --num-samples 10000 --output active_params.json
"""

import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import json
import os


def count_feedforward_params(dim: int, mlp_dim: int) -> int:
    """Count parameters in a FeedForward block."""
    # LayerNorm: weight + bias
    layernorm = dim * 2
    # Linear(dim -> mlp_dim) + bias
    linear1 = dim * mlp_dim + mlp_dim
    # Linear(mlp_dim -> dim) + bias
    linear2 = mlp_dim * dim + dim
    return layernorm + linear1 + linear2


def count_attention_params(dim: int, heads: int, dim_head: int) -> int:
    """Count parameters in an Attention block (Flash or regular)."""
    inner_dim = heads * dim_head
    # LayerNorm: weight + bias
    layernorm = dim * 2
    # to_qkv: Linear(dim -> inner_dim * 3), no bias
    to_qkv = dim * inner_dim * 3
    # to_out: Linear(inner_dim -> dim) + bias
    to_out = inner_dim * dim + dim
    return layernorm + to_qkv + to_out


def count_model_params(vocab_size: int, max_seq_len: int, dim: int, depth: int,
                       heads: int, dim_head: int, mlp_dim: int,
                       moe_layers: list, num_experts: int) -> dict:
    """
    Count parameters broken down by component.
    
    Returns a dict with:
        - total_params: Total parameters in the model
        - base_params: Parameters always active (embeddings, attention, non-MoE FFN)
        - moe_params_per_layer: Parameters in each MoE layer (all experts)
        - moe_params_per_expert: Parameters per expert (same for all)
    """
    # Token embedding (output projection shares weights via weight tying)
    token_embedding = vocab_size * dim
    
    # Positional embedding
    pos_embedding = max_seq_len * dim
    
    # Final LayerNorm in transformer
    final_layernorm = dim * 2
    
    # Output projection shares weights with token_embedding, so 0 additional
    output_projection = 0
    
    # Attention params per layer
    attn_per_layer = count_attention_params(dim, heads, dim_head)
    total_attention = attn_per_layer * depth
    
    # FFN params per layer
    ffn_per_layer = count_feedforward_params(dim, mlp_dim)
    
    # Non-MoE layers
    non_moe_layers = [i for i in range(depth) if i not in moe_layers]
    non_moe_ffn_total = ffn_per_layer * len(non_moe_layers)
    
    # MoE layers - each has num_experts copies of FFN plus gating
    # Gating params per MoE layer:
    #   - GAMoEGateT.sim_matrix: (max_expert_num=64, dim) but only num_experts rows used conceptually
    #   - Actually stored as (dim, max_expert_num) = dim * 64 floats
    #   - GAMoEGateT.gates: max_expert_num = 64 floats
    #   - GAMoEGateT.temperature: 1 float (but typically not counted as it's fixed)
    # For simplicity, we'll use the actual stored size
    max_expert_num = 64  # hardcoded in GAMoEGateT
    gating_params_per_layer = dim * max_expert_num + max_expert_num
    
    moe_ffn_per_layer = ffn_per_layer * num_experts + gating_params_per_layer
    moe_total = moe_ffn_per_layer * len(moe_layers)
    
    # Base params (always active)
    base_params = (token_embedding + pos_embedding + final_layernorm + 
                   output_projection + total_attention + non_moe_ffn_total)
    
    # For MoE layers, gating is always active, but experts are conditional
    gating_total = gating_params_per_layer * len(moe_layers)
    base_params += gating_total
    
    return {
        'total_params': base_params + ffn_per_layer * num_experts * len(moe_layers),
        'base_params': base_params,
        'ffn_per_expert': ffn_per_layer,
        'num_moe_layers': len(moe_layers),
        'num_experts': num_experts,
        'moe_layers': moe_layers,
        'breakdown': {
            'token_embedding': token_embedding,
            'pos_embedding': pos_embedding,
            'final_layernorm': final_layernorm,
            'attention_total': total_attention,
            'non_moe_ffn_total': non_moe_ffn_total,
            'gating_total': gating_total,
            'moe_experts_total': ffn_per_layer * num_experts * len(moe_layers),
        }
    }


def compute_active_params(param_info: dict, avg_experts_per_token: dict) -> dict:
    """
    Compute average active parameters given expert activation statistics.
    
    Args:
        param_info: Output from count_model_params()
        avg_experts_per_token: Dict mapping layer_name -> average experts per token
                               e.g., {"layer_4": 2.3, "layer_5": 1.8, ...}
    
    Returns:
        Dict with active parameter statistics
    """
    base_params = param_info['base_params']
    ffn_per_expert = param_info['ffn_per_expert']
    num_experts = param_info['num_experts']
    moe_layers = param_info['moe_layers']
    
    # Calculate active MoE params
    active_moe_params = 0
    per_layer_stats = {}
    
    for layer_idx in moe_layers:
        layer_name = f"layer_{layer_idx}"
        if layer_name in avg_experts_per_token:
            avg_k = avg_experts_per_token[layer_name]
        else:
            # Default to 1 if not found
            avg_k = 1.0
            print(f"Warning: No expert stats for {layer_name}, using avg_k=1.0")
        
        # Active params for this layer = avg_k * params_per_expert
        layer_active = avg_k * ffn_per_expert
        active_moe_params += layer_active
        
        per_layer_stats[layer_name] = {
            'avg_experts_per_token': avg_k,
            'active_params': int(layer_active),
            'total_params': ffn_per_expert * num_experts,
            'activation_ratio': avg_k / num_experts,
        }
    
    total_active = base_params + active_moe_params
    total_params = param_info['total_params']
    
    # Overall average k across all MoE layers
    if avg_experts_per_token:
        overall_avg_k = np.mean([v for k, v in avg_experts_per_token.items() 
                                 if k.startswith('layer_')])
    else:
        overall_avg_k = 1.0
    
    return {
        'total_params': int(total_params),
        'total_params_billions': total_params / 1e9,
        'active_params': int(total_active),
        'active_params_billions': total_active / 1e9,
        'base_params': int(base_params),
        'base_params_billions': base_params / 1e9,
        'active_moe_params': int(active_moe_params),
        'activation_ratio': total_active / total_params,
        'overall_avg_experts_per_token': float(overall_avg_k),
        'per_layer_stats': per_layer_stats,
    }


def get_args():
    parser = argparse.ArgumentParser(description="Compute average active parameters for MoE model")
    
    # Checkpoint args
    parser.add_argument("--checkpoint-dir", type=str, required=True, dest="checkpoint_dir",
                        help="Directory containing checkpoints")
    parser.add_argument("--checkpoint-step", type=int, required=True, dest="checkpoint_step",
                        help="Checkpoint step to load (e.g., 124000)")
    
    # Data args
    parser.add_argument("--data-dir", type=str, required=True, dest="data_dir",
                        help="Path to tokenized data directory")
    parser.add_argument("--num-samples", type=int, default=10000, dest="num_samples",
                        help="Number of samples to process for statistics (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                        help="Batch size for inference")
    
    # Model architecture args (should match training)
    parser.add_argument("--vocab-size", type=int, dest="vocab_size", default=131072)
    parser.add_argument("--max-seq-len", type=int, dest="max_seq_len", default=2048)
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--dim-head", type=int, dest="dim_head", default=128)
    parser.add_argument("--mlp-dim", type=int, dest="mlp_dim", default=12288)
    parser.add_argument("--moe-layers", type=int, nargs='+', dest="moe_layers",
                        default=list(range(4, 28)))
    parser.add_argument("--num-experts", type=int, dest="num_experts", default=8)
    parser.add_argument("--ep-size", type=int, dest="ep_size", default=8)
    parser.add_argument("--topany-gating-impl", type=str, dest="topany_gating_impl", default="sparse")
    parser.add_argument("--use-flash-attention", action="store_true", dest="use_flash_attention")
    parser.add_argument("--l1-lambda", type=float, dest="l1_lambda", default=0.0001)
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results (optional)")
    
    return parser.parse_args()


def run_inference_for_stats(args):
    """Load model and run inference to collect expert activation statistics."""
    import deepspeed
    from htyllm_pg.model_builder import moe_builder
    from htyllm_pg.dataset import MultiLangTokenDataset
    from torch.utils.data import DataLoader, Subset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Build model
    print("Building model...")
    model = moe_builder(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        moe_layers=args.moe_layers,
        num_experts=args.num_experts,
        ep_size=args.ep_size,
        topany_gating_impl=args.topany_gating_impl,
        use_flash_attention=args.use_flash_attention,
        use_gradient_checkpointing=False,  # Not needed for inference
        l1_lambda=args.l1_lambda,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from step {args.checkpoint_step}...")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"step_{args.checkpoint_step}")
    
    # For DeepSpeed checkpoints, we need to handle the sharded format
    # First try to load as a DeepSpeed checkpoint
    try:
        # Initialize DeepSpeed for inference
        ds_config = {
            "train_batch_size": args.batch_size,
            "fp16": {"enabled": True},
        }
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        model_engine.load_checkpoint(args.checkpoint_dir, tag=f"step_{args.checkpoint_step}")
        model = model_engine.module
        print("Loaded DeepSpeed checkpoint successfully")
    except Exception as e:
        print(f"Could not load as DeepSpeed checkpoint: {e}")
        print("Trying to load as PyTorch state_dict...")
        # Try loading consolidated checkpoint
        state_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(state_path):
            model.load_state_dict(torch.load(state_path, map_location=device))
            print("Loaded PyTorch state_dict successfully")
        else:
            raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = MultiLangTokenDataset(args.data_dir, seq_length=args.max_seq_len)
    
    # Subset for efficiency
    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.permutation(len(dataset))[:num_samples]
    subset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Collect statistics
    print(f"Running inference on {num_samples} samples...")
    expert_activations = defaultdict(list)  # layer_name -> list of avg_k per batch
    total_tokens = 0
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Processing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            _, _, expert_counts = model(input_ids, attention_mask=attention_mask)
            
            # Count valid tokens
            if attention_mask is not None:
                num_valid_tokens = attention_mask.sum().item()
            else:
                num_valid_tokens = input_ids.numel()
            
            total_tokens += num_valid_tokens
            
            # Collect expert activation stats
            for layer_name, exp_counts in expert_counts.items():
                if exp_counts is not None:
                    # exp_counts: [num_experts] - count of tokens per expert
                    total_activations = exp_counts.sum().item()
                    avg_k = total_activations / num_valid_tokens if num_valid_tokens > 0 else 0
                    expert_activations[layer_name].append(avg_k)
    
    # Average across all batches
    avg_experts_per_token = {}
    for layer_name, k_values in expert_activations.items():
        avg_experts_per_token[layer_name] = np.mean(k_values)
    
    print(f"\nProcessed {total_tokens:,} tokens total")
    print("\nAverage experts per token by layer:")
    for layer_name in sorted(avg_experts_per_token.keys(), key=lambda x: int(x.split('_')[1])):
        print(f"  {layer_name}: {avg_experts_per_token[layer_name]:.3f}")
    
    return avg_experts_per_token


def main():
    args = get_args()
    
    # Calculate parameter breakdown
    print("=" * 60)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 60)
    
    param_info = count_model_params(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        moe_layers=args.moe_layers,
        num_experts=args.num_experts,
    )
    
    print(f"\nModel Configuration:")
    print(f"  dim={args.dim}, depth={args.depth}, heads={args.heads}")
    print(f"  dim_head={args.dim_head}, mlp_dim={args.mlp_dim}")
    print(f"  vocab_size={args.vocab_size:,}, max_seq_len={args.max_seq_len}")
    print(f"  MoE layers: {len(args.moe_layers)} layers ({min(args.moe_layers)}-{max(args.moe_layers)})")
    print(f"  num_experts={args.num_experts}")
    
    print(f"\nParameter Breakdown:")
    for key, value in param_info['breakdown'].items():
        print(f"  {key}: {value:,} ({value/1e6:.2f}M)")
    
    print(f"\nTotal Parameters: {param_info['total_params']:,} ({param_info['total_params']/1e9:.3f}B)")
    print(f"Base Parameters (always active): {param_info['base_params']:,} ({param_info['base_params']/1e9:.3f}B)")
    
    # Run inference to get actual statistics
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE FOR EXPERT ACTIVATION STATISTICS")
    print("=" * 60)
    avg_experts_per_token = run_inference_for_stats(args)
    
    # Compute active parameters
    print("\n" + "=" * 60)
    print("ACTIVE PARAMETER ANALYSIS")
    print("=" * 60)
    
    results = compute_active_params(param_info, avg_experts_per_token)
    
    print(f"\nOverall Average Experts Per Token: {results['overall_avg_experts_per_token']:.3f}")
    print(f"\nParameter Summary:")
    print(f"  Total Parameters:  {results['total_params']:,} ({results['total_params_billions']:.3f}B)")
    print(f"  Base Parameters:   {results['base_params']:,} ({results['base_params_billions']:.3f}B)")
    print(f"  Active MoE Params: {results['active_moe_params']:,} ({results['active_moe_params']/1e9:.3f}B)")
    print(f"  Total Active:      {results['active_params']:,} ({results['active_params_billions']:.3f}B)")
    print(f"\nActivation Ratio: {results['activation_ratio']:.2%}")
    print(f"  (i.e., on average, {results['activation_ratio']*100:.1f}% of parameters are active per token)")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
