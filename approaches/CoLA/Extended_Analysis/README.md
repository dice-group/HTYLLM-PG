# Extended Analysis: Expert Routing Analysis for CoLA/HydraLoRA

Expert routing analysis toolkit for understanding how language-specific adapters route to different experts.

## Overview

This toolkit analyzes expert routing patterns in CoLA and HydraLoRA adapter checkpoints. It allows you to:

### Expected Patterns
- Layer-wise specialization (low→high entropy from early to late layers)
- Language family clustering in routing space
- Expert concentration in later layers
- Related languages routing to similar experts

## Quick Start

### 1. Prepare Test Data

Extract 10,000 sequences per language from your dataset:

```bash
python tool/prepare_language_datasets.py \
    --validation_data /path/to/samples \
    --languages "en,es,hi,ru,fi" \
    --num_sequences 10000 \
    --output_dir ./data/language_test_sets
```

**Note**: Use simple language codes (en, es, hi, etc.). The script automatically maps them to actual directory names (`english`, `spa_Latn`, `hin_Deva`, etc.).

### 2. Run Expert Routing Analysis

Analyze a trained checkpoint:

```bash
python tool/analyze_expert_routing.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_checkpoint /path/to/checkpoint \
    --adapter_type hydralora \
    --test_data ./data/language_test_sets \
    --output ./analysis/my_checkpoint \
    --num_layers 32 \
    --num_experts 4
```

### 3. Process and Normalize Data

Apply critical layer-wise normalization:

```bash
python tool/process_routing_data.py \
    --input ./analysis/my_checkpoint/routing_matrix.npz \
    --output ./analysis/my_checkpoint/routing_matrix_normalized.npz
```

### 4. Generate Visualizations

Create heatmaps, t-SNE plots, and entropy analysis:

```bash
python tool/visualize_expert_routing.py \
    --routing_data ./analysis/my_checkpoint/routing_matrix_normalized.npz \
    --language_families ./config/language_families.json \
    --output_dir ./analysis/my_checkpoint/figures \
    --create_all
```

### 5. Generate Report

Create comprehensive analysis report:

```bash
python tool/generate_analysis_report.py \
    --routing_data ./analysis/my_checkpoint/routing_matrix_normalized.npz \
    --language_families ./config/language_families.json \
    --figures_dir ./analysis/my_checkpoint/figures \
    --output ./analysis/my_checkpoint/report.md
```

## Complete Pipeline

### Local/Interactive Execution

```bash
./run_analysis_pipeline.sh
```

### HPC/Slurm Execution

**Option 1: Single Job (Sequential)**
```bash
sbatch run_analysis_pipeline_SLURM.sh \
    meta-llama/Llama-3.1-8B \
    /path/to/checkpoint \
    hydralora \
    /path/to/samples \
    "en,es,hi,ru,fi"
```

**Option 2: Parallel Job Array (Faster)**
```bash
# Step 1: Run per-language analysis in parallel
sbatch --array=0-4 run_per_language_SLURM.sh \
    meta-llama/Llama-3.1-8B \
    /path/to/checkpoint \
    hydralora \
    ./data/language_test_sets \
    ./analysis/my_checkpoint

# Step 2: Wait for all jobs to complete, then aggregate
./aggregate_results.sh ./analysis/my_checkpoint
```

### Manual Step-by-Step

```bash
# 1. Prepare data
python tool/prepare_language_datasets.py \
    --validation_data /path/to/samples \
    --languages "en,es,hi,ru" \
    --num_sequences 10000 \
    --output_dir ./data/language_test_sets

# 2. Analyze routing
python tool/analyze_expert_routing.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_checkpoint /path/to/checkpoint \
    --adapter_type hydralora \
    --test_data ./data/language_test_sets \
    --output ./analysis/my_checkpoint

# 3. Normalize data
python tool/process_routing_data.py \
    --input $OUTPUT_DIR/routing_matrix.npz \
    --output $OUTPUT_DIR/routing_matrix_normalized.npz

# 4. Visualize
python tool/visualize_expert_routing.py \
    --routing_data $OUTPUT_DIR/routing_matrix_normalized.npz \
    --language_families ./config/language_families.json \
    --output_dir $OUTPUT_DIR/figures \
    --create_all

# 5. Generate report
python tool/generate_analysis_report.py \
    --routing_data $OUTPUT_DIR/routing_matrix_normalized.npz \
    --language_families ./config/language_families.json \
    --figures_dir $OUTPUT_DIR/figures \
    --output $OUTPUT_DIR/report.md

echo "✓ Analysis complete! Report: $OUTPUT_DIR/report.md"
```

## Tools Reference

### prepare_language_datasets.py
Extracts language-specific test sets from dataset.

**Key Arguments**:
- `--validation_data`: Path to samples directory with language subdirectories
- `--languages`: Simple language codes (e.g., `"en,es,hi,ru"`)
- `--num_sequences`: Sequences per language (default: 10000)
- `--text_field`: Field name for text content (default: `"text"`)

**Data Format**: Expects directories like `english/`, `spa_Latn/`, `hin_Deva/` containing `.jsonl` or `.jsonl.gz` files.

### analyze_expert_routing.py
Main analysis script that captures routing decisions during inference.

**Key Arguments**:
- `--base_model`: Base model name/path
- `--adapter_checkpoint`: Adapter checkpoint directory
- `--adapter_type`: `cola` or `hydralora`
- `--num_layers`: Number of model layers (32 for Llama-7B)
- `--num_experts`: Experts per layer

### Normalization (Critical!)

Layer-wise normalization formula:.

**Critical**: This normalization is essential for correct visualization!

### visualize_expert_routing.py
Generates visualizations.

**Outputs**:
- `routing_heatmap.png` - Figure 10 replica
- `tsne_clustering.png` - Language family clustering
- `layer_entropy.png` - Specialization across layers

### generate_analysis_report.py
Creates comprehensive markdown report with embedded figures.

## Directory Structure

```
Extended_Analysis/
├── tool/                           # Analysis scripts
│   ├── prepare_language_datasets.py
│   ├── analyze_expert_routing.py
│   ├── process_routing_data.py
│   ├── visualize_expert_routing.py
│   └── generate_analysis_report.py
├── config/
│   └── language_families.json      # Language taxonomy (63 languages)
├── run_analysis_pipeline.sh        # Local execution
├── run_analysis_pipeline_SLURM.sh  # Slurm single job
├── run_per_language_SLURM.sh       # Slurm job array
├── aggregate_results.sh            # Merge parallel results
├── data/
│   └── language_test_sets/         # Prepared test data
└── analysis/                       # Analysis outputs
    └── checkpoint_name/
        ├── raw_stats/              # Per-language raw data
        ├── routing_matrix.npz      # Raw routing counts
        ├── routing_matrix_normalized.npz  # Normalized data
        ├── metadata.json
        ├── figures/
        │   ├── routing_heatmap.png
        │   ├── tsne_clustering.png
        │   └── layer_entropy.png
        └── report.md               # Analysis report
```

## Key Implementation Details

### Layer-Wise Normalization

```python
normalized = count / (total_tokens / num_layers)
```

This is the critical formula for proper visualization across layers.

### Visualization
- **Heatmap**: Blue-white-yellow-red color scheme with layer boundaries
- **t-SNE**: Dimensionality reduction with learning_rate=250, init='pca', perplexity=30

```python
TSNE(
    learning_rate=250.0,  # Higher than default
    init='pca',           # PCA initialization
    perplexity=30.0
)
```

### Color Scheme

Blue → White → Yellow → Red with breakpoints at 1/16 and 2/16 of num_experts.

## Requirements

```bash
pip install numpy matplotlib seaborn scikit-learn torch transformers peft datasets tqdm
```

## Troubleshooting

### No routing hooks attached
- Check that `adapter_type` matches your checkpoint (cola vs hydralora)
- Verify the adapter is properly loaded

### Empty routing statistics
- Ensure your adapter layers expose routing information via forward hooks
- Check that the model is in eval mode

### Heatmap looks wrong
- Verify you ran `process_routing_data.py` to apply normalization
- Use the normalized `.npz` file for visualization

## References

The analysis methodology is inspired by techniques from multilingual MoE research:

```bibtex
@article{laurenccon2024lola,
  title={LOLA: Large and Open Source Multilingual Language Model},
  author={Laurençon, Hugo and others},
  journal={arXiv preprint},
  year={2024}
}
```

Our implementation is original code designed specifically for CoLA/HydraLoRA adapters.

## Support

For issues or questions, refer to the main implementation plan:
- `../../.gemini/antigravity/brain/.../lola_expert_analysis_plan.md`
