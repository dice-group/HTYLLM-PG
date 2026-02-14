# CoLA: Hierarchical Asymmetric Adapter Pipeline

This folder contains our hierarchical multilingual adapter work based on CoLA and HydraLoRA, including language-aware routing for massively multilingual training.

## What This Approach Investigates

- Hierarchical asymmetric adapters (shared low-rank structure + language-specific components)
- CoLA and HydraLoRA routing variants
- Language Prior Routing (LPR) with language-id guidance
- Multilingual ablations across 200 languages

## Folder Guide

- `LLaMA-Factory/`: training framework with CoLA/HydraLoRA implementation changes
- `data_prep/`: data sampling, clustering, tokenizer extension, and tokenization pipeline
- `scripts/`: SLURM/local launchers for training and evaluation
- `configs/`: evaluation task lists and run configuration inputs
- `tools/two_stage_clustering/`: language grouping JSONs
- `docs/`: full documentation and generated PDF/MD builds
- `result_analysis/`: evaluation exports and analysis scripts

## Setup

1. Initialize submodules from the repository root:
```bash
git submodule update --init --recursive
```

2. Create and activate the conda environment:
```bash
cd approaches/CoLA/LLaMA-Factory
conda env create -f environment.yml
conda activate merlin
```

3. Install LLaMA-Factory in editable mode so local CoLA/Hydra changes are used:
```bash
pip uninstall -y peft llamafactory
pip install -e .
```

 follow `approaches/CoLA/LLaMA-Factory/setup_conda_env.md` for more details

4. Check model/data paths and environment variables in the SLURM scripts before launching (cluster-specific `/scratch/...` paths are referenced in several scripts).

## Running

From `approaches/CoLA/`:

- Baseline CoLA training:
```bash
cd scripts
sbatch accelerate_moe_cola_train.sh
```

- Multilingual ablation launcher:
```bash
cd scripts/comparison
sbatch run_multilingual_ablation.sh
```

- Single-variant comparison jobs:
  - `scripts/comparison/cola_lpr_job.sh`
  - `scripts/comparison/hydralora_lpr_job.sh`
  - `scripts/comparison/lora_job.sh`

## Documentation (Start Here)

- `docs/README.md`
- `docs/01_project_documentation.md`
- `docs/02_data_preparation.md`
- `docs/03_model_training_and_implementation.md`
- `docs/04_training_orchestration.md`
- `docs/05_evaluation_and_analysis.md`
- `docs/06_reproducibility_and_submission.md`

Deep-dive references:

- `docs/extra/hierarchical_adapters_multilingual_study_approaches_explanation.md`
- `docs/extra/storyline.md`
