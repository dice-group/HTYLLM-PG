#!/usr/bin/env python3
"""
Standalone Group Analysis Tool for Expert Routing

Aggregates existing routing analysis by language families or subgroups.
Does NOT require re-running model inference - works with existing routing_matrix.npz

Usage:
    python analyze_by_groups.py \
        --input /path/to/routing_matrix_normalized.npz \
        --groupings /path/to/200_tier_language_groupings.json \
        --output /path/to/output_dir \
        --aggregate_by families  # or 'subgroups'
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_groupings(groupings_file: Path) -> Dict:
    """Load language groupings configuration."""
    logger.info(f"Loading groupings from {groupings_file}")
    with open(groupings_file, 'r') as f:
        groupings = json.load(f)
    logger.info(f"Loaded {len(groupings)} top-level groups")
    return groupings


def aggregate_routing_by_families(
    routing_matrix: np.ndarray,
    languages: List[str],
    groupings: Dict
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate routing data by language families (top-level groups).
    
    IMPORTANT: This function AVERAGES the routing probabilities across languages
    within each family. This is correct because:
    1. Input routing_matrix is already layer-normalized (each layer sums to ~1.0)
    2. Averaging gives us the representative routing pattern for the family
    3. Summing would artificially favor larger families (incorrect)
    
    Args:
        routing_matrix: [num_languages, num_layers, num_experts] - MUST be normalized
        languages: List of language codes
        groupings: Language groupings from 200_tier_language_groupings.json
    
    Returns:
        aggregated_matrix: [num_families, num_layers, num_experts]
        family_labels: List of family IDs with counts
    """
    logger.info("Aggregating by families (top-level groups)")
    
    num_layers = routing_matrix.shape[1]
    num_experts = routing_matrix.shape[2]
    
    # Build language -> family mapping
    lang_to_family = {}
    for family_id, family_data in groupings.items():
        for lang in family_data['languages']:
            lang_to_family[lang] = family_id
    
    # Get sorted family IDs
    family_ids = sorted(groupings.keys(), key=lambda x: int(x))
    num_families = len(family_ids)
    
    # Create aggregated matrix
    aggregated = np.zeros((num_families, num_layers, num_experts), dtype=np.float32)
    family_counts = np.zeros(num_families, dtype=np.int32)
    
    # Aggregate
    for lang_idx, lang in enumerate(languages):
        if lang not in lang_to_family:
            logger.warning(f"Language '{lang}' not found in groupings, skipping")
            continue
        
        family_id = lang_to_family[lang]
        family_idx = family_ids.index(family_id)
        aggregated[family_idx] += routing_matrix[lang_idx]
        family_counts[family_idx] += 1
    
    # Average (not sum) to get representative routing pattern
    for family_idx in range(num_families):
        if family_counts[family_idx] > 0:
            aggregated[family_idx] /= family_counts[family_idx]
    
    # Create labels with counts
    family_labels = [
        f"Family_{fid} (n={family_counts[i]})" 
        for i, fid in enumerate(family_ids)
    ]
    
    logger.info(f"Aggregated {len(languages)} languages into {num_families} families")
    logger.info(f"Family sizes: {dict(zip(family_ids, family_counts))}")
    
    return aggregated, family_labels


def aggregate_routing_by_subgroups(
    routing_matrix: np.ndarray,
    languages: List[str],
    groupings: Dict
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate routing data by subgroups (B-level groups).
    
    IMPORTANT: This function AVERAGES the routing probabilities across languages
    within each subgroup. This is correct because:
    1. Input routing_matrix is already layer-normalized (each layer sums to ~1.0)
    2. Averaging gives us the representative routing pattern for the subgroup
    3. Summing would artificially favor larger subgroups (incorrect)
    
    Args:
        routing_matrix: [num_languages, num_layers, num_experts] - MUST be normalized
        languages: List of language codes
        groupings: Language groupings from 200_tier_language_groupings.json
    
    Returns:
        aggregated_matrix: [num_subgroups, num_layers, num_experts]
        subgroup_labels: List of subgroup IDs with counts
    """
    logger.info("Aggregating by subgroups (B-level)")
    
    num_layers = routing_matrix.shape[1]
    num_experts = routing_matrix.shape[2]
    
    # Build language -> (family, subgroup) mapping
    lang_to_subgroup = {}
    all_subgroups = []
    
    for family_id, family_data in groupings.items():
        for subgroup_id, subgroup_langs in family_data['subgroups'].items():
            full_subgroup_id = f"{family_id}_{subgroup_id}"
            all_subgroups.append(full_subgroup_id)
            for lang in subgroup_langs:
                lang_to_subgroup[lang] = full_subgroup_id
    
    # Sort subgroups
    all_subgroups = sorted(set(all_subgroups), key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    num_subgroups = len(all_subgroups)
    
    # Create aggregated matrix
    aggregated = np.zeros((num_subgroups, num_layers, num_experts), dtype=np.float32)
    subgroup_counts = np.zeros(num_subgroups, dtype=np.int32)
    
    # Aggregate
    for lang_idx, lang in enumerate(languages):
        if lang not in lang_to_subgroup:
            logger.warning(f"Language '{lang}' not found in subgroups, skipping")
            continue
        
        subgroup_id = lang_to_subgroup[lang]
        subgroup_idx = all_subgroups.index(subgroup_id)
        aggregated[subgroup_idx] += routing_matrix[lang_idx]
        subgroup_counts[subgroup_idx] += 1
    
    # Average
    for subgroup_idx in range(num_subgroups):
        if subgroup_counts[subgroup_idx] > 0:
            aggregated[subgroup_idx] /= subgroup_counts[subgroup_idx]
    
    # Create labels with counts
    subgroup_labels = [
        f"{sgid} (n={subgroup_counts[i]})" 
        for i, sgid in enumerate(all_subgroups)
    ]
    
    logger.info(f"Aggregated {len(languages)} languages into {num_subgroups} subgroups")
    
    return aggregated, subgroup_labels


def create_routing_heatmap(
    routing_matrix: np.ndarray,
    group_labels: List[str],
    num_layers: int,
    num_experts: int,
    output_file: Path,
    color_scheme: str = 'modern'
):
    """Create routing heatmap for grouped data with modern color scheme."""
    logger.info(f"Creating routing heatmap for {len(group_labels)} groups with {color_scheme} color scheme")
    
    # Reshape for visualization
    # [num_groups, num_layers, num_experts] -> [num_groups, num_layers*num_experts]
    heatmap_data = routing_matrix.reshape(len(group_labels), -1)
    
    # Create custom color scheme matching visualize_expert_routing.py
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
    
    # Create figure
    if color_scheme == 'modern':
        fig, ax = plt.subplots(figsize=(25, max(10, len(group_labels) * 0.4)))
    else:
        fig, ax = plt.subplots(figsize=(20, max(8, len(group_labels) * 0.3)))
    
    # Plot heatmap
    im = ax.imshow(
        heatmap_data,
        cmap=custom_cmap,
        aspect='auto',
        vmin=0,
        vmax=1,
        interpolation='nearest'
    )
    
    # Add colorbar with tick marks
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Routing Ratio (normalized per layer)', fontsize=12, fontweight='bold')
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    
    # Labels
    ax.set_yticks(range(len(group_labels)))
    ax.set_yticklabels(group_labels, fontsize=10)
    ax.set_xlabel('Experts (grouped by layer)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Language Groups', fontsize=14, fontweight='bold')
    ax.set_title(
        'Expert Routing Patterns: Grouped Language-Expert Associations Across Layers',
        fontsize=16,
        fontweight='bold'
    )
    
    # Add vertical lines to separate layers
    for i in range(1, num_layers):
        ax.axvline(x=i * num_experts - 0.5, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_file}")


def generate_report(
    aggregated_matrix: np.ndarray,
    group_labels: List[str],
    groupings: Dict,
    aggregate_by: str,
    output_file: Path,
    has_target_routing: bool = False
):
    """Generate analysis report."""
    logger.info("Generating analysis report")
    
    num_groups, num_layers, num_experts = aggregated_matrix.shape
    
    # Per-group expert preferences
    group_preferences = []
    for group_idx, group_label in enumerate(group_labels):
        expert_usage = aggregated_matrix[group_idx].sum(axis=0)  # Sum across layers
        top_expert = np.argmax(expert_usage)
        top_usage = expert_usage[top_expert]
        group_preferences.append((group_label, top_expert, top_usage))
    
    # Generate markdown report
    visualizations = """## Visualizations

- **Actual Routing Heatmap**: `figures/grouped_routing_heatmap.png`"""
    
    if has_target_routing:
        visualizations += "\n- **Target Routing Heatmap**: `figures/grouped_target_routing_heatmap.png`"
    
    report = f"""# Expert Routing Analysis by {aggregate_by.capitalize()}

## Summary

- **Aggregation Level**: {aggregate_by}
- **Number of Groups**: {num_groups}
- **Number of Layers**: {num_layers}
- **Number of Experts**: {num_experts}

## Group Expert Preferences

Top expert by total routing probability:

| Group | Top Expert | Usage % |
|-------|-----------|---------|
"""
    
    for group_label, top_expert, top_usage in sorted(group_preferences, key=lambda x: -x[2]):
        usage_pct = (top_usage / aggregated_matrix.shape[1]) * 100  # Normalize by layers
        report += f"| {group_label} | Expert {top_expert} | {usage_pct:.1f}% |\n"
    
    report += f"""

{visualizations}

## Analysis Notes

This analysis aggregates routing patterns by {aggregate_by}. Each group's routing pattern 
is the **average** of all languages within that group. The heatmap shows which experts are 
preferred by each group across all layers.

For detailed per-language analysis, see the original routing analysis.
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved report to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze expert routing by language groups/families',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--input', type=Path, required=True,  help='Path to routing_matrix_normalized.npz')
    parser.add_argument('--groupings', type=Path, required=True, help='Path to 200_tier_language_groupings.json')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for grouped analysis')
    parser.add_argument('--aggregate_by', choices=['families', 'subgroups'], default='families', help='Aggregate by top-level families or subgroups (default: families)')
    parser.add_argument('--color_scheme', choices=['modern', 'classic'], default='modern', help='Color scheme for heatmap: modern (white-green-yellow-red-black) or classic (blue-white-yellow-red)')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading routing data from {args.input}")
    data = np.load(args.input)
    
    routing_matrix = data['routing_matrix']
    languages = list(data['languages'])
    num_layers = int(data['num_layers'])
    num_experts = int(data['num_experts'])
    
    logger.info(f"Loaded routing matrix: {routing_matrix.shape}")
    logger.info(f"Languages: {len(languages)}")
    
    # Check for target routing
    has_target_routing = bool(data.get('has_target_routing', False))
    target_routing_matrix = None
    router_mode = str(data.get('router_mode', 'unknown'))
    if has_target_routing and 'target_routing_matrix' in data:
        target_routing_matrix = data['target_routing_matrix']
        logger.info(f"Found target routing matrix: {target_routing_matrix.shape} (mode: {router_mode})")
    
    # Load groupings
    groupings = load_groupings(args.groupings)
    
    # Aggregate actual routing
    if args.aggregate_by == 'families':
        aggregated_matrix, group_labels = aggregate_routing_by_families(
            routing_matrix, languages, groupings
        )
    else:  # subgroups
        aggregated_matrix, group_labels = aggregate_routing_by_subgroups(
            routing_matrix, languages, groupings
        )
    
    # Aggregate target routing if available
    aggregated_target_matrix = None
    if target_routing_matrix is not None:
        logger.info("Aggregating target routing matrix")
        if args.aggregate_by == 'families':
            aggregated_target_matrix, _ = aggregate_routing_by_families(
                target_routing_matrix, languages, groupings
            )
        else:  # subgroups
            aggregated_target_matrix, _ = aggregate_routing_by_subgroups(
                target_routing_matrix, languages, groupings
            )
    
    # Create output directories
    args.output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Save aggregated data
    output_npz = args.output / 'grouped_routing_matrix.npz'
    save_dict = {
        'routing_matrix': aggregated_matrix,
        'group_labels': np.array(group_labels, dtype=object),
        'num_layers': num_layers,
        'num_experts': num_experts,
        'aggregate_by': args.aggregate_by,
        'has_target_routing': has_target_routing,
        'router_mode': router_mode
    }
    
    if aggregated_target_matrix is not None:
        save_dict['target_routing_matrix'] = aggregated_target_matrix
        logger.info("Including aggregated target routing in output")
    
    np.savez(output_npz, **save_dict)
    logger.info(f"Saved aggregated data to {output_npz}")
    
    # Generate heatmap visualization
    create_routing_heatmap(
        aggregated_matrix,
        group_labels,
        num_layers,
        num_experts,
        figures_dir / 'grouped_routing_heatmap.png',
        color_scheme=args.color_scheme
    )
    
    # Generate target routing heatmap if available
    if aggregated_target_matrix is not None:
        # Create filename based on mode
        if router_mode == 'hard':
            target_filename = 'grouped_enforced_routing_heatmap.png'
            logger.info("Creating grouped enforced routing heatmap (hard mode)")
        else:
            target_filename = 'grouped_target_routing_heatmap.png'
            logger.info(f"Creating grouped target routing heatmap ({router_mode} mode)")
        
        create_routing_heatmap(
            aggregated_target_matrix,
            group_labels,
            num_layers,
            num_experts,
            figures_dir / target_filename,
            color_scheme=args.color_scheme
        )
    
    # Generate report
    generate_report(
        aggregated_matrix,
        group_labels,
        groupings,
        args.aggregate_by,
        args.output / 'grouped_analysis_report.md',
        has_target_routing=has_target_routing
    )
    
    logger.info("=" * 60)
    logger.info("Group Analysis Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"- Aggregated data: grouped_routing_matrix.npz")
    logger.info(f"- Actual routing heatmap: figures/grouped_routing_heatmap.png")
    if has_target_routing:
        if router_mode == 'hard':
            logger.info(f"- Enforced routing heatmap: figures/grouped_enforced_routing_heatmap.png")
        else:
            logger.info(f"- Target routing heatmap: figures/grouped_target_routing_heatmap.png")
    logger.info(f"- Report: grouped_analysis_report.md")


if __name__ == '__main__':
    main()
