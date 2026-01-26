#!/usr/bin/env python3
"""
Visualize expert routing patterns with heatmaps and t-SNE clustering.

Usage:
    python visualize_expert_routing.py \
        --routing_data ./analysis/checkpoint-5000/routing_matrix_normalized.npz \
        --language_families ../config/language_families.json \
        --output_dir ./analysis/checkpoint-5000/figures \
        --create_all
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_language_families(families_file: Path) -> Dict:
    """Load language family mappings."""
    with open(families_file, 'r') as f:
        return json.load(f)


def create_routing_heatmap(
    routing_matrix: np.ndarray,
    languages: List[str],
    num_layers: int,
    num_experts: int,
    output_file: Path,
    color_scheme: str = 'modern'
):
    """
    Create heatmap with white-green-yellow-red-black color scheme.
    
    routing_matrix: numpy array [num_languages, num_layers, num_experts]
                    MUST be normalized using layer-wise normalization!
    languages: list of language codes
    color_scheme: 'classic' (blue-white-yellow-red) or 'modern' (white-green-yellow-red-black)
    """
    logger.info(f"Creating routing heatmap with {color_scheme} color scheme")
    
    # Reshape to [languages, layers*experts]
    heatmap_data = routing_matrix.reshape(len(languages), -1)
    
    # white-green-yellow-red-black color scheme
    if color_scheme == 'modern':
        # white-green-yellow-red-black (newer, more vibrant)
        colors = ['white', 'lightgreen', 'yellow', 'lightcoral', 'red', 'black']
        # Ensure positions are strictly increasing by capping 4/experts at 0.7 max
        pos_4 = min(4.0/num_experts, 0.7)
        positions = [0, 1.0/num_experts, 2.0/num_experts, pos_4, 0.75, 1.0]
        tick_positions = [0, 1.0/num_experts, 2.0/num_experts, pos_4, 0.5, 1.0]
        tick_labels = ['0', f'1/{num_experts}', f'2/{num_experts}', f'4/{num_experts}', '1/2', '1']
    else:  # classic
        # blue-white-yellow-red
        colors = ['blue', 'white', 'yellow', 'red', 'red']
        positions = [0, 1.0/num_experts, 2.0/num_experts, 0.75, 1.0]
        tick_positions = [0, 1.0/num_experts, 2.0/num_experts, 0.5, 1.0]
        tick_labels = ['0', f'1/{num_experts}', f'2/{num_experts}', '1/2', '1']
    
    custom_cmap = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)))
    
    # Create figure (4000x1700 for heatmap2, 1600x1200 for classic)
    if color_scheme == 'modern':
        fig, ax = plt.subplots(figsize=(25, 10.625))  # 16:9 aspect ratio
    else:
        fig, ax = plt.subplots(figsize=(20, 12))
    
    # Heatmap with color scheme and range
    im = ax.imshow(
        heatmap_data,
        cmap=custom_cmap,
        aspect='auto',
        vmin=0,  # [0, 1] range for normalized data
        vmax=1,
        interpolation='nearest'
    )
    
    # Add vertical lines to separate layers
    for layer in range(1, num_layers):
        ax.axvline(x=layer * num_experts - 0.5, color='black', linewidth=2)
    
    # Labels
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(languages, fontsize=8)  # Smaller font for many languages
    ax.set_xlabel('Experts (grouped by layer)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Language', fontsize=14, fontweight='bold')
    ax.set_title(
        'Expert Routing Patterns: Language-Expert Associations Across Layers',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Colorbar with tick marks
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Routing Ratio (normalized per layer)', fontsize=12)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    
    # Add layer labels at top
    for layer in range(num_layers):
        x_pos = layer * num_experts + num_experts // 2
        ax.text(
            x_pos, -1,
            f'L{layer}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_file}")


def create_tsne_clustering(
    routing_matrix: np.ndarray,
    languages: List[str],
    language_families: Dict,
    output_file: Path
):
    """
    Create t-SNE visualization for language clustering.
    
    Parameters: learning_rate=250, init='pca', perplexity=30
    """
    logger.info("Creating t-SNE clustering visualization")
    
    # Flatten per language
    if routing_matrix.ndim == 3:
        routing_matrix_flat = routing_matrix.reshape(len(routing_matrix), -1)
    else:
        routing_matrix_flat = routing_matrix
    
    # Replace NaN with 0
    routing_matrix_flat = np.nan_to_num(routing_matrix_flat, nan=0.0)
    
    # Validate data before t-SNE
    n_samples = len(routing_matrix_flat)
    
    # Check for all zeros
    if np.allclose(routing_matrix_flat, 0):
        logger.error("Routing matrix is all zeros - cannot perform t-SNE")
        logger.error("This suggests no routing data was collected. Check that:")
        logger.error("  1. Your model has MoE/routing layers")
        logger.error("  2. Forward hooks attached correctly")
        logger.error("  3. Inference ran successfully")
        # Create empty plot with error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 
                'ERROR: No routing data available\n(all values are zero)',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Check for constant values (no variance)
    if np.std(routing_matrix_flat) < 1e-10:
        logger.error("Routing matrix has no variance - cannot perform t-SNE")
        logger.error("All routing values are identical")
        # Create empty plot with error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 
                'ERROR: No variance in routing data\n(all values are identical)',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Adaptive perplexity based on number of samples
    # t-SNE requires perplexity < n_samples
    perplexity = min(30.0, max(2.0, n_samples - 1))
    
    if n_samples < 5:
        logger.warning(f"Very few samples ({n_samples}), t-SNE may not be meaningful")
    
    logger.info(f"Using perplexity={perplexity} for {n_samples} languages")
    logger.info(f"Data range: [{routing_matrix_flat.min():.4f}, {routing_matrix_flat.max():.4f}]")
    logger.info(f"Data std: {routing_matrix_flat.std():.4f}")
    
    # t-SNE parameters
    logger.info("Running t-SNE (this may take a minute)...")
    try:
        tsne = TSNE(
            n_components=2,
            learning_rate=250.0,
            init='pca',
            perplexity=perplexity,
            random_state=42,
            max_iter=1000  # Limit iterations to prevent hanging
        )
        
        embedded = tsne.fit_transform(routing_matrix_flat.astype(np.float32))
    except Exception as e:
        logger.error(f"t-SNE failed: {e}")
        # Create error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 
                f'ERROR: t-SNE computation failed\n{str(e)}',
                ha='center', va='center', fontsize=12, color='red',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Get language family info
    lang_family_map = language_families.get('languages', {})
    family_colors = language_families.get('family_colors', {})
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Group by family
    families = {}
    for i, lang in enumerate(languages):
        lang_info = lang_family_map.get(lang, {})
        family = lang_info.get('family', 'Unknown')
        if family not in families:
            families[family] = []
        families[family].append(i)
    
    # Plot each family with different color
    for family, indices in families.items():
        color = family_colors.get(family, '#808080')
        x = embedded[indices, 0]
        y = embedded[indices, 1]
        ax.scatter(
            x, y,
            label=family,
            color=color,
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Annotate points with language codes
        for i, idx in enumerate(indices):
            ax.annotate(
                languages[idx],
                (embedded[idx, 0], embedded[idx, 1]),
                fontsize=8,
                alpha=0.8,
                xytext=(3, 3),
                textcoords='offset points'
            )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(
        'Language Clustering by Expert Routing Patterns',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        fontsize=9
    )
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved t-SNE plot to {output_file}")


def create_layer_entropy_plot(
    layer_entropy: np.ndarray,
    output_file: Path
):
    """Plot expert specialization (entropy) across layers."""
    logger.info("Creating layer entropy plot")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = np.arange(len(layer_entropy))
    ax.plot(layers, layer_entropy, marker='o', linewidth=2, markersize=8)
    ax.fill_between(layers, layer_entropy, alpha=0.3)
    
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Expert Specialization Across Layers\n(Lower entropy = higher specialization)',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved entropy plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize expert routing patterns"
    )
    parser.add_argument(
        '--routing_data',
        type=Path,
        required=True,
        help='Normalized routing matrix (.npz file)'
    )
    parser.add_argument(
        '--language_families',
        type=Path,
        required=True,
        help='Language families JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for figures'
    )
    parser.add_argument(
        '--create_heatmap',
        action='store_true',
        help='Create routing heatmap'
    )
    parser.add_argument(
        '--create_tsne',
        action='store_true',
        help='Create t-SNE clustering plot'
    )
    parser.add_argument(
        '--create_entropy',
        action='store_true',
        help='Create layer entropy plot'
    )
    parser.add_argument(
        '--color_scheme',
        type=str,
        choices=['classic', 'modern'],
        default='modern',
        help='Heatmap color scheme: classic (blue-white-yellow-red) or modern (white-green-yellow-red-black)'
    )
    parser.add_argument(
        '--create_all',
        action='store_true',
        help='Create all visualizations'
    )
    
    args = parser.parse_args()
    
    # If create_all, enable all visualizations
    if args.create_all:
        args.create_heatmap = True
        args.create_tsne = True
        args.create_entropy = True
    
    # Load data
    logger.info(f"Loading routing data from {args.routing_data}")
    data = np.load(args.routing_data)
    
    routing_matrix = data['routing_matrix']
    languages = list(data['languages'])
    num_layers = int(data['num_layers'])
    num_experts = int(data['num_experts'])
    
    # Check for target routing
    has_target_routing = bool(data.get('has_target_routing', False))
    target_routing_matrix = None
    router_mode = str(data.get('router_mode', 'unknown'))
    if has_target_routing and 'target_routing_matrix' in data:
        target_routing_matrix = data['target_routing_matrix']
        logger.info(f"Found target routing matrix for dual heatmap mode (mode: {router_mode})")
    
    logger.info(f"Matrix shape: {routing_matrix.shape}")
    logger.info(f"Languages: {len(languages)}")
    
    # Load language families
    language_families = load_language_families(args.language_families)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.create_heatmap:
        # Router learning heatmap (what the router network learned/selected)
        create_routing_heatmap(
            routing_matrix,
            languages,
            num_layers,
            num_experts,
            args.output_dir / 'routing_heatmap.png',
            color_scheme=args.color_scheme
        )
        
        # Target routing heatmap (LPR targets)
        # For hard mode: what's enforced
        # For learned/bias modes: what LPR supervises toward
        if target_routing_matrix is not None:
            # Create filename based on mode
            if router_mode == 'hard':
                target_filename = 'enforced_routing_heatmap.png'
                logger.info("Creating enforced routing heatmap (hard mode)")
            else:
                target_filename = 'target_routing_heatmap.png'
                logger.info(f"Creating target routing heatmap ({router_mode} mode)")
            
            create_routing_heatmap(
                target_routing_matrix,
                languages,
                num_layers,
                num_experts,
                args.output_dir / target_filename,
                color_scheme=args.color_scheme
            )
            logger.info(f"Generated both router learning and target routing heatmaps")
    
    if args.create_tsne:
        create_tsne_clustering(
            routing_matrix,
            languages,
            language_families,
            args.output_dir / 'tsne_clustering.png'
        )
    
    if args.create_entropy and 'layer_entropy' in data:
        create_layer_entropy_plot(
            data['layer_entropy'],
            args.output_dir / 'layer_entropy.png'
        )
    
    logger.info("Visualization complete!")


if __name__ == '__main__':
    main()
