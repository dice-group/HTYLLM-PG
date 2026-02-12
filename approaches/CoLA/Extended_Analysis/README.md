# Extended Analysis: Expert Routing Analysis for CoLA/HydraLoRA

Expert routing analysis toolkit for understanding how language-specific adapters route to different experts in Mixture-of-Experts (MoE) architectures.

## Overview

This toolkit analyzes expert routing patterns in **CoLA** and **HydraLoRA** adapter checkpoints. It captures routing decisions during inference, normalizes the data, and generates comprehensive visualizations and reports to understand:

- **Actual vs. Target Routing**: Compare learned routing behavior with configured language-expert assignments
- **Layer-wise Specialization**: Analyze entropy changes across model layers (low→high from early to late layers)
- **Language Family Clustering**: Visualize how related languages route similarly using t-SNE
- **Expert Concentration**: Identify which experts are preferred per language and layer
- **Router Mode Analysis**: Support for CoLA's hard routing, learned routing, and bias modes

### Key Features

✅ **Dual Routing Visualization**: Shows both actual routing (what the router network learned) and target routing (what it was configured to do)  
✅ **Multi-Adapter Support**: Works with CoLA and HydraLoRA checkpoints  
✅ **Auto-Detection**: Automatically detects adapter type, num_layers, and num_experts from config  
✅ **Router Mode Aware**: Handles different routing modes (hard, learned, bias) with appropriate visualizations  
✅ **Modern Visualization**: White-green-yellow-red-black color scheme optimized for normalized routing probabilities  

## Quick Start

### 1. Prepare Test Data

Extract test sequences per language from your dataset:

```bash
python tool/prepare_language_datasets.py \
    --validation_data /path/to/samples \
    --languages "acm_Arab,ary_Arab,bel_Cyrl,hin_Deva,eng_Latn" \
    --num_sequences 10000 \
    --output_dir ./data/language_test_sets
```

### 2. Run Expert Routing Analysis

Analyze a trained checkpoint (adapter_type, num_layers, num_experts auto-detected):

```bash
python tool/analyze_expert_routing.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_checkpoint /path/to/checkpoint_adapter \
    --test_data ./data/language_test_sets \
    --output ./analysis/my_checkpoint \
    --batch_size 16 \
    --max_sequences 100 \
    --use_language_ids
```

**Important Flags**:
- `--use_language_ids`: Pass language IDs to the model (required for hard routing modes)
- `--max_sequences`: Limit sequences per language to control runtime

### 3. Process and Normalize Data

Apply critical layer-wise normalization:

```bash
python tool/process_routing_data.py \
    --input ./analysis/my_checkpoint/routing_matrix.npz \
    --output ./analysis/my_checkpoint/routing_matrix_normalized.npz
```

**What This Does**:
- Applies layer-wise normalization: `normalized = count / (total_tokens / num_layers)`
- Ensures each layer's routing sums to ~1.0 for fair comparison
- Preserves target routing matrix if present

### 4. Generate Visualizations

Create heatmaps, t-SNE plots, and entropy analysis:

```bash
python tool/visualize_expert_routing.py \
    --routing_data ./analysis/my_checkpoint/routing_matrix_normalized.npz \
    --language_families ./config/language_families.json \
    --output_dir ./analysis/my_checkpoint/figures \
    --create_all \
    --color_scheme modern
```

**Outputs**:
- `routing_heatmap.png` - Actual routing patterns (what the router learned)
- `target_routing_heatmap.png` or `enforced_routing_heatmap.png` - Expected routing (from config)
- `tsne_clustering.png` - Language family clustering by routing similarity
- `layer_entropy.png` - Routing specialization across layers

### 5. Generate Report

Create comprehensive analysis report with embedded figures:

```bash
python tool/generate_analysis_report.py \
    --routing_data ./analysis/my_checkpoint/routing_matrix_normalized.npz \
    --language_families ./config/language_families.json \
    --figures_dir ./analysis/my_checkpoint/figures \
    --output ./analysis/my_checkpoint/report.md
```

## Complete Pipeline Scripts

### HPC/Slurm Execution (Recommended)

```bash
# Configure paths in run_analysis_pipeline_SLURM.sh (lines 51-64):
# - BASE_MODEL
# - CHECKPOINT (must end with _adapter)
# - VALIDATION_DATA
# - LANGUAGES
# - NUM_SEQUENCES
# - BATCH_SIZE

sbatch run_analysis_pipeline_SLURM.sh
```

**Resource Requirements**:
- **Time**: 12 hours
- **GPU**: 1x H100 (or equivalent)
- **Memory**: 256GB RAM
- **CPUs**: 32 cores

**Output Structure**:
```
/scratch/.../extended_analysis/<variant_name>/<checkpoint_name>/
├── routing_matrix.npz
├── routing_matrix_normalized.npz
├── metadata.json
├── figures/
│   ├── routing_heatmap.png
│   ├── target_routing_heatmap.png (or enforced_routing_heatmap.png)
│   ├── tsne_clustering.png
│   └── layer_entropy.png
└── report.md
```

**Output Directory Naming**:
- `<variant_name>`: Parent folder of checkpoint (e.g., `cola_colaexp-hard_20260108_054502`)
- `<checkpoint_name>`: Checkpoint folder name (e.g., `checkpoint-50000_adapter`)
- Example: `/scratch/.../cola_colaexp-hard_20260108_054502/checkpoint-50000_adapter/`

### Local/Interactive Execution

```bash
./run_analysis_pipeline.sh
```

Edit the script to configure paths and parameters.

## Tools Reference

### prepare_language_datasets.py

Extracts language-specific test sets from dataset directory.

**Key Arguments**:
- `--validation_data`: Path to samples directory with language subdirectories
- `--languages`: Comma-separated ISO 639-3+script codes (e.g., `"eng_Latn,spa_Latn,hin_Deva"`)
- `--num_sequences`: Sequences per language (default: 10000)
- `--text_field`: JSON field name for text content (default: `"text"`)
- `--output_dir`: Where to save extracted test sets

**Expected Data Format**:
```
validation_data/
├── acm_Arab/
│   └── data.jsonl.gz
├── eng_Latn/
│   └── data.jsonl.gz
└── ...
```

### analyze_expert_routing.py

Main analysis script that captures routing decisions during inference.

**Key Arguments**:
- `--base_model`: Base model name/path (e.g., `meta-llama/Llama-3.1-8B`)
- `--adapter_checkpoint`: Adapter checkpoint directory (must end with `_adapter`)
- `--test_data`: Directory with language test sets
- `--output`: Output directory for analysis results
- `--batch_size`: Batch size for inference (default: 16)
- `--max_sequences`: Max sequences per language (default: 100)
- `--device`: Device to use (default: cuda)
- `--use_language_ids`: Pass language IDs to model (required for hard routing)

**Auto-Detected Parameters** (from `adapter_config.json`):
- `--adapter_type`: CoLA or HydraLoRA
- `--num_layers`: Number of model layers
- `--num_experts`: Number of experts per layer

**What It Does**:
1. Loads base model and adapter checkpoint
2. Attaches forward hooks to capture routing logits
3. Runs inference on test data
4. Builds routing matrix `[num_languages, num_layers, num_experts]`
5. Generates target routing matrix from `language_to_family_ids` if present
6. Saves raw routing counts and metadata

**Output Files**:
- `routing_matrix.npz`: Raw routing counts
  - `routing_matrix`: `[langs, layers, experts]` token counts
  - `languages`: List of language codes
  - `num_layers`, `num_experts`: Dimensions
  - `has_target_routing`: Boolean flag
  - `router_mode`: `"hard"`, `"learned"`, or `"bias"`
  - `target_routing_matrix`: Binary matrix `[langs, layers, experts]` (if applicable)
- `metadata.json`: Checkpoint info, parameters, statistics

**Routing Capture Details**:
- **CoLA**: Captures from `cola_router_logits` cache key (after softmax)
- **HydraLoRA**: Captures from `hydra_expert_router_logits` cache key
- **Legacy**: Falls back to `.routing_logits` attribute if caches unavailable

### process_routing_data.py

Applies layer-wise normalization to routing counts.

**Key Arguments**:
- `--input`: Raw routing matrix file (`routing_matrix.npz`)
- `--output`: Normalized output file (`.npz`)

**Normalization Formula**:
```python
normalized[lang, layer, expert] = count / (total_tokens / num_layers)
```

**Why This Matters**:
- Without normalization, early layers dominate the heatmap (more tokens processed)
- Layer-wise normalization ensures each layer contributes equally
- Enables fair comparison across layers

**Preserved Data**:
- Target routing matrix (already 0/1, no normalization needed)
- Router mode metadata
- Language family information

### visualize_expert_routing.py

Generates visualizations from normalized routing data.

**Key Arguments**:
- `--routing_data`: Normalized routing matrix (`.npz`)
- `--language_families`: JSON file with language taxonomy
- `--output_dir`: Directory for generated figures
- `--color_scheme`: `"modern"` (default) or `"classic"`
- `--create_heatmap`: Generate routing heatmap only
- `--create_tsne`: Generate t-SNE plot only
- `--create_entropy`: Generate layer entropy plot only
- `--create_all`: Generate all visualizations

**Visualizations**:

#### 1. Routing Heatmap (`routing_heatmap.png`)
- **What**: Actual routing probabilities learned by the router network
- **Dimensions**: Languages × (Layers × Experts)
- **Color Scheme**: 
  - **Modern** (default): white → lightgreen → yellow → lightcoral → red → black
  - **Classic**: blue → white → yellow → red
- **Features**:
  - Vertical black lines separate layers
  - Layer labels (L0, L1, ...) at top
  - Colorbar shows routing ratio with key thresholds (1/N, 2/N, 4/N, 1/2, 1)
  - Languages on Y-axis (labels hidden for clarity with many languages)

#### 2. Target/Enforced Routing Heatmap
- **Filename**:
  - `enforced_routing_heatmap.png` for **hard routing mode**
  - `target_routing_heatmap.png` for **learned/bias modes**
- **What**: Expected routing from `language_to_family_ids` configuration
- **Purpose**: Compare against actual routing to see how well the model follows the target
- **Only generated if**: Checkpoint has `language_to_family_ids` in `adapter_config.json`

#### 3. t-SNE Clustering (`tsne_clustering.png`)
- **What**: 2D projection of routing "fingerprints"
- **Method**: t-SNE with `learning_rate=250`, `init='pca'`, `perplexity=30`
- **Colors**: Language families from `language_families.json`
- **Interpretation**: Languages with similar routing patterns cluster together

#### 4. Layer Entropy Plot (`layer_entropy.png`)
- **What**: Shannon entropy of expert distribution per layer
- **Formula**: `H = -Σ p(expert) * log₂(p(expert))`
- **Interpretation**:
  - **Low entropy**: Routing concentrates on few experts (high specialization)
  - **High entropy**: Routing distributed across many experts (low specialization)
- **Expected Pattern**: Entropy increases from early to late layers (less → more specialization)

**Color Scheme Details** (Modern):
```python
positions = [0, 1/N, 2/N, 4/N, 0.75, 1.0]
colors = ['white', 'lightgreen', 'yellow', 'lightcoral', 'red', 'black']
tick_labels = ['0', '1/N', '2/N', '4/N', '1/2', '1']
```
Where N = num_experts. This emphasizes important thresholds:
- < 1/N: No routing (white)
- ~ 1/N: Equal distribution (lightgreen)
- 2/N - 4/N: Moderate preference (yellow → lightcoral)
- > 1/2: Strong preference (red → black)

### generate_analysis_report.py

Creates comprehensive markdown report with embedded visualizations.

**Key Arguments**:
- `--routing_data`: Normalized routing matrix
- `--language_families`: Language taxonomy JSON
- `--figures_dir`: Directory with generated figures
- `--output`: Output markdown file path

**Report Sections**:
1. **Summary Statistics**: Dimensions, entropy stats, language count
2. **Router Configuration**: Mode (hard/learned/bias), target routing status
3. **Expert Usage Analysis**: Top experts per language
4. **Visualizations**: Embedded heatmaps, t-SNE, entropy plots
5. **Interpretation Guide**: How to read the figures

## Directory Structure

```
Extended_Analysis/
├── tool/                              # Analysis scripts
│   ├── prepare_language_datasets.py   # Extract test data
│   ├── analyze_expert_routing.py      # Capture routing decisions
│   ├── process_routing_data.py        # Normalize routing data
│   ├── visualize_expert_routing.py    # Generate figures
│   └── generate_analysis_report.py    # Create markdown report
├── config/
│   └── language_families.json         # Language taxonomy for t-SNE coloring
├── run_analysis_pipeline.sh           # Local execution script
├── run_analysis_pipeline_SLURM.sh     # HPC/Slurm execution script
├── data/
│   └── language_test_sets/            # Prepared test data (generated)
├── analysis/                          # Analysis outputs (generated)
│   └── <variant>/<checkpoint>/
│       ├── routing_matrix.npz
│       ├── routing_matrix_normalized.npz
│       ├── metadata.json
│       ├── figures/
│       │   ├── routing_heatmap.png
│       │   ├── target_routing_heatmap.png  (or enforced_routing_heatmap.png)
│       │   ├── tsne_clustering.png
│       │   └── layer_entropy.png
│       └── report.md
├── logs/                              # SLURM job logs (generated)
│   ├── routing_analysis_<jobid>.out
│   └── routing_analysis_<jobid>.err
└── README.md                          # This file
```

## Key Implementation Details

### 1. Layer-Wise Normalization

**Formula**:
```python
normalized[lang, layer, expert] = count[lang, layer, expert] / (total_tokens[lang] / num_layers)
```

**Why**:
- Raw counts are higher in early layers (more tokens processed before filtering)
- Normalization ensures each layer contributes equally to visualization
- Each layer's routing sums to approximately 1.0

**Example**:
- Language processes 1000 total tokens across 32 layers
- Layer 0 processes 500 tokens (before filtering), layer 31 processes 200 tokens
- Normalizing by `1000/32 = 31.25` makes layers comparable

### 2. Routing Capture Mechanism

**Forward Hook Attachment**:
```python
def _make_hook(layer_idx):
    def hook(module, input, output):
        # Try cache-based capture (CoLA/HydraLoRA)
        if 'cola_router_logits' in module._caches:
            routing_probs = module._caches['cola_router_logits']
        elif 'hydra_expert_router_logits' in module._caches:
            routing_probs = module._caches['hydra_expert_router_logits']
        # Fall back to attribute-based capture
        elif hasattr(module, 'routing_logits'):
            routing_probs = module.routing_logits
        # ... process and store
    return hook
```

**Cache Keys**:
- **CoLA**: Uses `cola_router_logits` (set in `cola/forward.py`)
- **HydraLoRA**: Uses `hydra_expert_router_logits` (set in `hydralora/forward.py`)

### 3. Target Routing Matrix

**Source**: `language_to_family_ids` in `adapter_config.json`

**Example Config**:
```json
{
  "language_to_family_ids": {
    "eng_Latn": 0,
    "deu_Latn": 0,
    "fra_Latn": 0,
    "spa_Latn": 1,
    "ita_Latn": 1,
    ...
  }
}
```

**Matrix Construction**:
```python
target_routing_matrix = np.zeros((num_langs, num_layers, num_experts))
for lang_idx, lang in enumerate(languages):
    family_id = language_to_family_ids[lang]
    target_routing_matrix[lang_idx, :, family_id] = 1.0  # All layers → assigned expert
```

**Interpretation**:
- Binary matrix (0 or 1) indicating which expert each language should use
- Used for:
  - **Hard mode**: Enforced routing (what actually happens)
  - **Learned/Bias modes**: Target routing (what LPR loss supervises toward)

### 4. Router Mode Handling

| Mode | Description | Visualization |
|------|-------------|---------------|
| **hard** | Routing strictly enforced via masking | `enforced_routing_heatmap.png` |
| **learned** | Routing learned with LPR supervision | `target_routing_heatmap.png` |
| **bias** | Router logits biased toward targets | `target_routing_heatmap.png` |

**Mode Detection**:
```python
router_mode = adapter_config.get('language_router_mode', 'unknown')
```

### 5. Color Scheme Mathematics

**Purpose**: Emphasize meaningful routing thresholds.

**Modern Scheme** (default):
```python
# For N experts:
breakpoints = [0, 1/N, 2/N, min(4/N, 0.7), 0.75, 1.0]
colors = ['white', 'lightgreen', 'yellow', 'lightcoral', 'red', 'black']
```

**Why These Positions**:
- **0 (white)**: No routing to this expert
- **1/N (lightgreen)**: Uniform distribution (no preference)
- **2/N (yellow)**: 2× uniform (moderate preference)
- **4/N (lightcoral)**: 4× uniform (strong preference)
- **0.5 (red)**: Half of all routing goes here
- **1.0 (black)**: All routing concentrated here

**Capping at 0.7**: With few experts (e.g., 4), `4/4=1.0` would conflict with final position. Capping ensures strictly increasing positions.

## Requirements

```bash
# Python packages
pip install numpy matplotlib seaborn scikit-learn torch transformers datasets tqdm

# Important: Do NOT pip install peft
# The local version in ../LLaMA-Factory/src/peft is used automatically
```

**System Requirements**:
- Python 3.8+
- CUDA-capable GPU (for inference)
- ~256GB RAM (for large-scale analysis)
- Disk space: ~10GB per checkpoint analysis

## Troubleshooting

### Issue: "No routing data captured"

**Cause**: Forward hooks not attaching or routing data not cached.

**Solutions**:
1. Verify adapter type matches checkpoint (`--adapter_type cola` or `hydralora`)
2. Check that adapter is loaded correctly (should see adapter layers in model)
3. Ensure model is in eval mode (automatically done by script)
4. For HydraLoRA, verify it's using the updated version with cache support

### Issue: "Target routing matrix not found"

**Cause**: Checkpoint doesn't have `language_to_family_ids` in `adapter_config.json`.

**Solutions**:
- This is normal for checkpoints without language-expert assignments
- Only CoLA with hard routing or LPR-trained models have this
- Visualization will proceed without target/enforced heatmap

### Issue: "Heatmap looks wrong / all one color"

**Cause**: Forgot to run normalization step.

**Solution**:
```bash
python tool/process_routing_data.py \
    --input ./analysis/output/routing_matrix.npz \
    --output ./analysis/output/routing_matrix_normalized.npz

# Then use routing_matrix_normalized.npz for visualization
```

### Issue: "Language not found in groupings"

**Cause**: Test language not in `language_families.json`.

**Solutions**:
1. Add missing language to `config/language_families.json`
2. Use only languages present in the config file
3. Warning is logged but analysis continues (language excluded from t-SNE)

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce `--batch_size` (try 8, 4, or 2)
2. Reduce `--max_sequences` per language
3. Use a GPU with more memory
4. Process fewer languages at once

### Issue: "Empty routing statistics for some languages"

**Cause**: Not enough data or model filtered all tokens.

**Solutions**:
1. Increase `--num_sequences` in test data preparation
2. Check that test data files actually contain data
3. Verify language codes match between data and config

## Advanced Usage

### Analyzing Multiple Checkpoints

```bash
# Method 1: Loop over checkpoints
for checkpoint in /path/to/checkpoints/checkpoint-*_adapter; do
    sbatch run_analysis_pipeline_SLURM.sh \
        meta-llama/Llama-3.1-8B \
        "$checkpoint" \
        ...
done

# Method 2: Use SLURM job arrays
# Create checkpoint_list.txt with one checkpoint path per line
# Then modify SLURM script to read from array
```

### Custom Language Sets

**Option 1**: Modify script configuration
```bash
LANGUAGES="eng_Latn,fra_Latn,deu_Latn,spa_Latn"  # Edit in .sh file
```

**Option 2**: Add new language families to config
```json
// config/language_families.json
{
  "Indo-European": {
    "color": "#FF6B6B",
    "languages": ["eng_Latn", "deu_Latn", "fra_Latn", ...]
  }
}
```

### Comparing Hard vs. Learned Routing

1. Analyze checkpoint with `language_router_mode: "hard"`
2. Analyze checkpoint with `language_router_mode: "learned"`
3. Compare:
   - Hard mode: `enforced_routing_heatmap.png` should match `routing_heatmap.png` exactly
   - Learned mode: `target_routing_heatmap.png` vs `routing_heatmap.png` shows supervision vs. learned behavior

## References

This analysis methodology is inspired by and designed to replicate techniques from:

```bibtex
@article{laurenccon2024lola,
  title={LOLA: Large Language Models as Open-source Assembler for Adapters},
  author={Laurençon, Hugo and others},
  journal={arXiv preprint arXiv:2402.xxxxx},
  year={2024}
}
```
