"""
Analyze expert usage patterns by language for a trained MoE model.

Usage:
    python -m htyllm_pg.analyze_expert_usage \
        --checkpoint-dir /path/to/checkpoints \
        --checkpoint-tag step_124000 \
        --data-dir /path/to/tokenized_multilingual \
        --output-dir /path/to/output \
        --samples-per-lang 500
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from htyllm_pg.model_builder import moe_builder


def load_model_from_checkpoint(checkpoint_dir, checkpoint_tag, config_path=None):
    """Load model from DeepSpeed checkpoint."""
    # Load config
    if config_path is None:
        config_path = os.path.join(checkpoint_dir, "config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Loaded config from {config_path}")
    print(f"Model config: dim={config['dim']}, depth={config['depth']}, "
          f"num_experts={config['num_experts']}, moe_layers={config['moe_layers']}")
    
    # Build the model
    model = moe_builder(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        dim_head=config["dim_head"],
        dropout=config.get("dropout", 0.0),
        emb_dropout=config.get("emb_dropout", 0.0),
        moe_layers=config["moe_layers"],
        num_experts=config["num_experts"],
        k=config.get("k", -1),
        capacity_factor=config.get("capacity_factor", 1.5),
        eval_capacity_factor=config.get("eval_capacity_factor", 2.0),
        min_capacity=config.get("min_capacity", 0.0),
        use_residual=config.get("use_residual", False),
        gate_backward=config.get("gate_backward", "ste"),
        ep_size=1,  # Single GPU for inference
        topany_gating_impl=config.get("topany_gating_impl", "sparse"),
        use_flash_attention=config.get("use_flash_attention", False),
        use_gradient_checkpointing=False,  # Not needed for inference
        l1_lambda=config.get("l1_lambda", 0.0005),
    )
    
    # Load checkpoint weights
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_tag)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # DeepSpeed checkpoints have a specific structure
    # For single-GPU loading, we need to load the model state dict directly
    mp_rank_file = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
    if os.path.exists(mp_rank_file):
        state_dict = torch.load(mp_rank_file, map_location="cpu")
        # DeepSpeed wraps the model state dict under 'module'
        if "module" in state_dict:
            model.load_state_dict(state_dict["module"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {mp_rank_file}")
    else:
        # Try loading from zero checkpoint format
        zero_file = os.path.join(checkpoint_path, "zero_pp_rank_0_mp_rank_00_model_states.pt")
        if os.path.exists(zero_file):
            state_dict = torch.load(zero_file, map_location="cpu")
            if "module" in state_dict:
                model.load_state_dict(state_dict["module"], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {zero_file}")
        else:
            raise FileNotFoundError(
                f"Could not find model states in {checkpoint_path}. "
                f"Checked: {mp_rank_file}, {zero_file}"
            )
    
    return model, config


def get_language_dirs(data_dir):
    """Get list of language directories in the tokenized data folder."""
    data_path = Path(data_dir)
    lang_dirs = []
    
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir():
            # Check if it has token files
            token_files = list(subdir.glob("tokens_*.npy"))
            if token_files:
                lang_dirs.append(subdir)
    
    return lang_dirs


def load_samples_for_language(lang_dir, num_samples, seq_length=2048):
    """Load a specified number of samples from a language directory."""
    token_files = sorted(lang_dir.glob("tokens_*.npy"))
    
    samples = []
    masks = []
    
    for token_file in token_files:
        if len(samples) >= num_samples:
            break
        
        mask_file = token_file.parent / token_file.name.replace("tokens_", "masks_")
        if not mask_file.exists():
            continue
        
        # Load with memory mapping
        tokens = np.load(token_file, mmap_mode='r')
        mask = np.load(mask_file, mmap_mode='r')
        
        # Take samples from this file
        remaining = num_samples - len(samples)
        n_from_file = min(len(tokens), remaining)
        
        for i in range(n_from_file):
            samples.append(tokens[i].astype(np.int64))
            masks.append(mask[i].astype(np.int64))
    
    if not samples:
        return None, None
    
    # Stack into tensors
    input_ids = torch.tensor(np.stack(samples))[:, :-1]  # Remove last token
    attention_mask = torch.tensor(np.stack(masks))[:, :-1]
    
    return input_ids, attention_mask


def analyze_expert_usage(model, data_dir, samples_per_lang, batch_size=8, device="cuda"):
    """Run inference and collect expert usage statistics per language."""
    model.eval()
    model.to(device)
    
    lang_dirs = get_language_dirs(data_dir)
    print(f"Found {len(lang_dirs)} language directories")
    
    # Results: {language: {layer: {expert: count}}}
    results = {}
    
    for lang_dir in tqdm(lang_dirs, desc="Processing languages"):
        lang_name = lang_dir.name
        
        input_ids, attention_mask = load_samples_for_language(
            lang_dir, samples_per_lang
        )
        
        if input_ids is None:
            print(f"  Skipping {lang_name}: no valid samples")
            continue
        
        # Initialize counters for this language
        lang_expert_counts = defaultdict(lambda: defaultdict(float))
        
        # Process in batches
        num_samples = len(input_ids)
        
        with torch.inference_mode():
            for i in range(0, num_samples, batch_size):
                batch_ids = input_ids[i:i+batch_size].to(device)
                batch_mask = attention_mask[i:i+batch_size].to(device)
                
                # Forward pass
                _, _, expert_counts = model(batch_ids, attention_mask=batch_mask)
                
                # Accumulate counts
                for layer_name, counts in expert_counts.items():
                    if counts is not None:
                        counts_np = counts.detach().cpu().numpy()
                        for expert_idx, count in enumerate(counts_np):
                            lang_expert_counts[layer_name][expert_idx] += float(count)
        
        results[lang_name] = dict(lang_expert_counts)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return results


def create_heatmap(results, output_path, num_experts=8):
    """Create a heatmap visualization of expert usage by language."""
    # Get sorted list of languages and layers
    languages = sorted(results.keys())
    
    # Get all layer names from first language that has data
    layer_names = []
    for lang in languages:
        if results[lang]:
            layer_names = sorted(
                results[lang].keys(), 
                key=lambda x: int(x.split('_')[1])
            )
            break
    
    if not layer_names:
        print("No layer data found!")
        return
    
    print(f"Creating heatmap for {len(languages)} languages, {len(layer_names)} layers")
    
    # Build the data matrix: rows = languages, cols = (layer, expert) pairs
    num_cols = len(layer_names) * num_experts
    data = np.zeros((len(languages), num_cols))
    
    for lang_idx, lang in enumerate(languages):
        for layer_idx, layer_name in enumerate(layer_names):
            if layer_name in results[lang]:
                total_for_layer = sum(results[lang][layer_name].values())
                for expert_idx in range(num_experts):
                    col_idx = layer_idx * num_experts + expert_idx
                    count = results[lang][layer_name].get(expert_idx, 0)
                    # Normalize to percentage
                    if total_for_layer > 0:
                        data[lang_idx, col_idx] = (count / total_for_layer) * 100
    
    # Create figure
    fig_width = max(20, len(layer_names) * 2)
    fig_height = max(10, len(languages) * 0.15)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=data.max())
    
    # Add vertical lines to separate layers
    for i in range(1, len(layer_names)):
        ax.axvline(x=i * num_experts - 0.5, color='black', linewidth=1)
    
    # Set y-axis labels (languages)
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(languages, fontsize=6)
    
    # Set x-axis labels (layer names at center of each layer's experts)
    layer_centers = [(i * num_experts + (num_experts - 1) / 2) for i in range(len(layer_names))]
    ax.set_xticks(layer_centers)
    ax.set_xticklabels([f"L{l.split('_')[1]}" for l in layer_names], fontsize=8)
    
    # Add expert number labels
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(range(num_cols))
    ax2.set_xticklabels([str(i % num_experts) for i in range(num_cols)], fontsize=6)
    ax2.set_xlabel("Expert Index")
    
    ax.set_xlabel("MoE Layer")
    ax.set_ylabel("Language")
    ax.set_title("Expert Usage (%) by Language and Layer")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Usage %")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def save_results_csv(results, output_path, num_experts=8):
    """Save results as a CSV file."""
    rows = []
    
    for lang, layer_data in results.items():
        for layer_name, expert_counts in layer_data.items():
            layer_idx = int(layer_name.split('_')[1])
            total = sum(expert_counts.values())
            
            for expert_idx in range(num_experts):
                count = expert_counts.get(expert_idx, 0)
                pct = (count / total * 100) if total > 0 else 0
                rows.append({
                    "language": lang,
                    "layer": layer_idx,
                    "layer_name": layer_name,
                    "expert": expert_idx,
                    "count": count,
                    "percentage": pct,
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert usage by language")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing the checkpoint")
    parser.add_argument("--checkpoint-tag", type=str, default="step_124000",
                        help="Checkpoint tag to load (e.g., step_124000)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing tokenized multilingual data")
    parser.add_argument("--output-dir", type=str, default="./expert_analysis",
                        help="Directory to save outputs")
    parser.add_argument("--samples-per-lang", type=int, default=500,
                        help="Number of samples to process per language")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to config.json (defaults to checkpoint-dir/config.json)")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize DeepSpeed distributed backend for single-GPU inference
    # This is required because DeepSpeed MoE layers use all_to_all communication
    import deepspeed
    if not torch.distributed.is_initialized():
        # Set environment variables for single-process distributed
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        
        deepspeed.init_distributed(dist_backend="nccl")
        print("Initialized DeepSpeed distributed backend")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint_dir}/{args.checkpoint_tag}...")
    model, config = load_model_from_checkpoint(
        args.checkpoint_dir, 
        args.checkpoint_tag,
        args.config_path
    )
    
    num_experts = config.get("num_experts", 8)
    
    # Analyze
    print(f"\nAnalyzing expert usage with {args.samples_per_lang} samples per language...")
    results = analyze_expert_usage(
        model, 
        args.data_dir, 
        args.samples_per_lang,
        batch_size=args.batch_size,
        device=device
    )
    
    # Save outputs
    heatmap_path = os.path.join(args.output_dir, "expert_usage_heatmap.png")
    csv_path = os.path.join(args.output_dir, "expert_usage_data.csv")
    json_path = os.path.join(args.output_dir, "expert_usage_raw.json")
    
    create_heatmap(results, heatmap_path, num_experts=num_experts)
    save_results_csv(results, csv_path, num_experts=num_experts)
    
    # Also save raw JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw JSON to {json_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
