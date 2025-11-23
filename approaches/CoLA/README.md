# Hierarchical Adapter Pipeline

Built by Yven & Sashreek and Joel to explore multilingual hierarchical CoLA/Hydra adapters with hierarchical (language-aware) routing

## Setup
1. **Conda env**: `cd LLaMA-Factory && conda env create -f environment.yaml && conda activate cola_llama_factory`.
2. **Local installs**: Afterwards uninstall peft and llamafactory again and `pip install -e .` (inside `LLaMA-Factory` and inside of `peft`).
3. **Models/data**: we use llama3.1B as well as llama3.2-1B / 3B. We have prepared/tokenized datasets referenced in the scripts (e.g. `/scratch/.../tokenized/...`). check on cluster for details

## Hierarchical Adapter TL;DR
CoLA/Hydra layers now share family-level A matrices, keep B/heads per language, and optionally use Language Prior Routing (bias/hard routing driven by batch-level language IDs + auxiliary loss). See `docs/storyline.md` for the full narrative and implementation details.

## Running (Slurm)
All launchers live in `scripts/`. For example, to train the standard Accelerate MoE CoLA baseline on the cluster:
```
cd scripts
sbatch accelerate_moe_cola_train.sh
```
Languag Prior routing is work in progress.
This should also be extended TODO