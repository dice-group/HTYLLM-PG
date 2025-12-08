#!/usr/bin/env python3
"""
Process routing data and apply layer-wise normalization.

This script loads raw routing data and applies the critical layer-wise
normalization formula to prepare data for visualization.

Usage:
    python process_routing_data.py \
        --input ./analysis/checkpoint-5000/routing_matrix.npz \
        --output ./analysis/checkpoint-5000/routing_matrix_normalized.npz
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_routing_counts(routing_counts: np.ndarray, num_layers: int) -> np.ndarray:
    """
    Normalize routing counts using layer-wise normalization.
    
    Args:
        routing_counts: [num_languages, num_layers, num_experts]
        num_layers: number of MoE layers
    
    Returns:
        normalized: same shape, normalized per layer
        
    Formula:
        normalized[lang, layer, expert] = count / (total_tokens_for_lang / num_layers)
    
    This ensures each layer's routing sums to ~1.0, preventing early layers
    from dominating the visualization.
    """
    logger.info("Applying layer-wise normalization")
    
    num_languages, num_layers, num_experts = routing_counts.shape
    normalized = np.zeros_like(routing_counts, dtype=np.float32)
    
    for lang_idx in range(num_languages):
        # Sum across all layers and experts for this language
        total_tokens = routing_counts[lang_idx].sum()
        
        if total_tokens == 0:
            logger.warning(f"Language {lang_idx} has zero tokens, skipping")
            continue
        
        # Normalize each layer
        for layer_idx in range(num_layers):
            layer_counts = routing_counts[lang_idx, layer_idx]
            # CRITICAL: divide by (total / num_layers), NOT just total!
            normalized[lang_idx, layer_idx] = layer_counts / (total_tokens / num_layers)
    
    logger.info("Normalization complete")
    return normalized


def compute_layer_entropy(normalized_matrix: np.ndarray) -> np.ndarray:
    """
    Compute entropy of expert distribution per layer.
    
    Higher entropy = more even distribution across experts
    Lower entropy = more specialization to specific experts
    
    Returns:
        entropy: [num_layers] array
    """
    logger.info("Computing layer-wise entropy")
    
    num_languages, num_layers, num_experts = normalized_matrix.shape
    entropy = np.zeros(num_layers)
    
    for layer_idx in range(num_layers):
        # Average distribution across all languages for this layer
        layer_dist = normalized_matrix[:, layer_idx, :].mean(axis=0)
        
        # Normalize to probability distribution
        layer_dist = layer_dist / (layer_dist.sum() + 1e-10)
        
        # Compute entropy: -sum(p * log(p))
        entropy[layer_idx] = -(layer_dist * np.log(layer_dist + 1e-10)).sum()
    
    return entropy


def main():
    parser = argparse.ArgumentParser(
        description="Process and normalize routing data"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input routing matrix (.npz file)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output normalized routing matrix (.npz file)'
    )
    
    args = parser.parse_args()
    
    # Load raw routing data
    logger.info(f"Loading routing data from {args.input}")
    data = np.load(args.input)
    
    routing_matrix = data['routing_matrix']
    languages = data['languages']
    num_layers = int(data['num_layers'])
    num_experts = int(data['num_experts'])
    
    logger.info(f"Loaded matrix shape: {routing_matrix.shape}")
    logger.info(f"Languages: {len(languages)}")
    logger.info(f"Layers: {num_layers}, Experts: {num_experts}")
    
    # Apply normalization
    normalized_matrix = normalize_routing_counts(routing_matrix, num_layers)
    
    # Compute additional statistics
    layer_entropy = compute_layer_entropy(normalized_matrix)
    
    # Save normalized data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        routing_matrix=normalized_matrix,
        routing_matrix_raw=routing_matrix,
        languages=languages,
        num_layers=num_layers,
        num_experts=num_experts,
        layer_entropy=layer_entropy
    )
    
    logger.info(f"Saved normalized data to {args.output}")
    logger.info("Processing complete!")


if __name__ == '__main__':
    main()
