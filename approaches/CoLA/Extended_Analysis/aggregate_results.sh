#!/bin/bash
# Aggregate results from parallel per-language jobs and generate final analysis

set -e

echo "Aggregating per-language results..."

# Configuration
OUTPUT_BASE="/analysis/checkpoint"
CONFIG_FILE="./config/language_families.json"

# Collect all per-language routing data
mkdir -p "${OUTPUT_BASE}/aggregated"

echo "Merging routing matrices..."
python - <<EOF
import numpy as np
import json
from pathlib import Path

output_base = Path("${OUTPUT_BASE}")
per_lang_dir = output_base / "per_language"
aggregated_dir = output_base / "aggregated"

# Find all language results
lang_dirs = sorted([d for d in per_lang_dir.iterdir() if d.is_dir()])

if not lang_dirs:
    print("No per-language results found!")
    exit(1)

print(f"Found {len(lang_dirs)} language results")

# Load first to get dimensions
first_data = np.load(lang_dirs[0] / "routing_matrix.npz")
num_layers = int(first_data['num_layers'])
num_experts = int(first_data['num_experts'])

# Collect all data
all_languages = []
all_matrices = []

for lang_dir in lang_dirs:
    lang = lang_dir.name
    data_file = lang_dir / "routing_matrix.npz"
    
    if not data_file.exists():
        print(f"Warning: No routing matrix for {lang}, skipping")
        continue
    
    data = np.load(data_file)
    all_languages.append(lang)
    all_matrices.append(data['routing_matrix'][0])  # First (and only) language

# Stack into single matrix
routing_matrix = np.stack(all_matrices, axis=0)
print(f"Aggregated matrix shape: {routing_matrix.shape}")

# Save aggregated data
np.savez(
    aggregated_dir / "routing_matrix.npz",
    routing_matrix=routing_matrix,
    languages=all_languages,
    num_layers=num_layers,
    num_experts=num_experts
)

print(f"✓ Saved aggregated matrix to {aggregated_dir}/routing_matrix.npz")

# Create metadata
metadata = {
    'num_languages': len(all_languages),
    'languages': all_languages,
    'num_layers': num_layers,
    'num_experts': num_experts,
    'routing_matrix_shape': list(routing_matrix.shape)
}

with open(aggregated_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved metadata")
EOF

# Now run the standard pipeline on aggregated data
echo ""
echo "Running normalization and visualization on aggregated data..."

python tool/process_routing_data.py \
    --input "${OUTPUT_BASE}/aggregated/routing_matrix.npz" \
    --output "${OUTPUT_BASE}/aggregated/routing_matrix_normalized.npz"

python tool/visualize_expert_routing.py \
    --routing_data "${OUTPUT_BASE}/aggregated/routing_matrix_normalized.npz" \
    --language_families "$CONFIG_FILE" \
    --output_dir "${OUTPUT_BASE}/aggregated/figures" \
    --create_all

python tool/generate_analysis_report.py \
    --routing_data "${OUTPUT_BASE}/aggregated/routing_matrix_normalized.npz" \
    --language_families "$CONFIG_FILE" \
    --figures_dir "${OUTPUT_BASE}/aggregated/figures" \
    --output "${OUTPUT_BASE}/aggregated/report.md"

echo ""
echo "========================================="
echo "✓ Aggregation Complete!"
echo "========================================="
echo "Final report: ${OUTPUT_BASE}/aggregated/report.md"
