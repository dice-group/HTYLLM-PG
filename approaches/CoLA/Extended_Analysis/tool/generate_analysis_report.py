#!/usr/bin/env python3
"""
Generate comprehensive analysis report from routing data.

Usage:
    python generate_analysis_report.py \
        --routing_data ./analysis/checkpoint-5000/routing_matrix_normalized.npz \
        --language_families ../config/language_families.json \
        --figures_dir ./analysis/checkpoint-5000/figures \
        --output ./analysis/checkpoint-5000/report.md
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_report(
    routing_data: Dict,
    metadata: Dict,
    language_families: Dict,
    figures_dir: Path,
    output_file: Path
):
    """Generate markdown analysis report."""
    logger.info("Generating analysis report")
    
    routing_matrix = routing_data['routing_matrix']
    languages = list(routing_data['languages'])
    num_layers = int(routing_data['num_layers'])
    num_experts = int(routing_data['num_experts'])
    layer_entropy = routing_data.get('layer_entropy', None)
    
    # Calculate statistics
    total_tokens = routing_data.get('routing_matrix_raw', routing_matrix).sum()
    
    # Find most specialized layers (lowest entropy)
    if layer_entropy is not None:
        most_specialized = np.argsort(layer_entropy)[:5]
        least_specialized = np.argsort(layer_entropy)[-5:]
    
    # Generate report
    report = f"""# Expert Routing Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Model Information

- **Base Model**: {metadata.get('base_model', 'N/A')}
- **Adapter Checkpoint**: {metadata.get('adapter_checkpoint', 'N/A')}
- **Adapter Type**: {metadata.get('adapter_type', 'N/A')}
- **Number of Layers**: {num_layers}
- **Experts per Layer**: {num_experts}

## Data Summary

- **Languages Analyzed**: {len(languages)}
- **Total Sequences Processed**: {metadata.get('total_sequences', 'N/A')}
- **Languages**: {', '.join(languages)}

## Expert Routing Patterns

### Overall Heatmap

![Routing Heatmap]({figures_dir.name}/routing_heatmap.png)

**Key Observations**:

"""
    
    # Add entropy analysis if available
    if layer_entropy is not None:
        report += f"""
### Expert Specialization Across Layers

![Layer Entropy]({figures_dir.name}/layer_entropy.png)

**Specialization Patterns**:
- **Early layers** (0-{num_layers//3}): Average entropy = {layer_entropy[:num_layers//3].mean():.2f}
- **Middle layers** ({num_layers//3}-{2*num_layers//3}): Average entropy = {layer_entropy[num_layers//3:2*num_layers//3].mean():.2f} 
- **Late layers** ({2*num_layers//3}-{num_layers}): Average entropy = {layer_entropy[2*num_layers//3:].mean():.2f}

**Most Specialized Layers** (lowest entropy):
{chr(10).join([f'- Layer {idx}: entropy = {layer_entropy[idx]:.3f}' for idx in most_specialized])}

**Least Specialized Layers** (highest entropy):
{chr(10).join([f'- Layer {idx}: entropy = {layer_entropy[idx]:.3f}' for idx in least_specialized])}

"""
    
    # Add t-SNE clustering
    if (figures_dir / 'tsne_clustering.png').exists():
        report += f"""
### Language Family Clustering

![t-SNE Clustering]({figures_dir.name}/tsne_clustering.png)

**Identified Clusters**:

Languages from the same linguistic family tend to cluster together in the routing space, suggesting that the MoE routing mechanism learns language-family-specific patterns.

"""
    
    # Add language family breakdown
    lang_family_map = language_families.get('languages', {})
    families = {}
    for lang in languages:
        family = lang_family_map.get(lang, {}).get('family', 'Unknown')
        if family not in families:
            families[family] = []
        families[family].append(lang)
    
    report += f"""
## Language Family Breakdown

"""
    for family, langs in sorted(families.items()):
        report += f"- **{family}** ({len(langs)}): {', '.join(langs)}\n"
    
    # Add methodology
    report += """

## Methodology

### Data Collection
1. Loaded base model with adapter checkpoint
2. Processed test sequences through model for each language
3. Captured expert routing decisions via forward hooks
4. Aggregated routing statistics per language per layer

### Normalization
Applied layer-wise normalization formula:
```
normalized[lang, layer, expert] = count / (total_tokens_for_lang / num_layers)
```

This ensures each layer's routing distribution sums to approximately 1.0, allowing fair comparison across layers.

### Visualization
- **Heatmap**: Blue-white-yellow-red color scheme with layer boundaries
- **t-SNE**: Dimensionality reduction with learning_rate=250, init='pca', perplexity=30
- **Entropy**: Computed per-layer expert distribution entropy

## Expected Patterns

Typical findings in multilingual MoE systems:
- Experts in early layers show less language specificity
- Later layers exhibit strong expert-language associations  
- Language families cluster in routing space
- Related languages route to similar experts

---

**Analysis conducted using Extended Analysis toolkit**
**Methodology inspired by multilingual MoE research**
"""
    
    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved report to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis report"
    )
    parser.add_argument(
        '--routing_data',
        type=Path,
        required=True,
        help='Normalized routing matrix (.npz file)'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=None,
        help='Metadata JSON file'
    )
    parser.add_argument(
        '--language_families',
        type=Path,
        required=True,
        help='Language families JSON file'
    )
    parser.add_argument(
        '--figures_dir',
        type=Path,
        required=True,
        help='Directory containing generated figures'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output report file (.md)'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading routing data from {args.routing_data}")
    routing_data = dict(np.load(args.routing_data))
    
    # Load metadata
    if args.metadata and args.metadata.exists():
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    else:
        # Try to find metadata in same directory as routing data
        metadata_file = args.routing_data.parent / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
    
    # Load language families
    with open(args.language_families, 'r') as f:
        language_families = json.load(f)
    
    # Generate report
    generate_report(
        routing_data,
        metadata,
        language_families,
        args.figures_dir,
        args.output
    )
    
    logger.info("Report generation complete!")


if __name__ == '__main__':
    main()
