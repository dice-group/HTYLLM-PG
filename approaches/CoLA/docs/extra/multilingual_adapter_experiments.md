# Multilingual Adapter Experiment Plan

This document outlines the ablations and run configurations for evaluating asymmetric LoRA variants (HydraLoRA, CoLA) with language-prior routing across model sizes and language subsets.

## Goals

1. Compare classical LoRA, flat HydraLoRA/CoLA, and routed variants that leverage Language Prior Routing (LPR) derived from MoE-LPR
2. Understand how routing + family-shared matrices scale with the number of languages sampled from `LANGUAGE_MAP` (`distributed_data_processor/language_subsets.py:1`).
3. Maintain fair training budgets across Llama-3 model sizes (`llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B`, `llama-3.2-3B_tokenizer|meta-llama/Llama-3.2-3B`, `llama-3.1-8B_tokenizer|meta-llama/Llama-3.1-8B`).

## Experiment Axes

### Model Sizes

- **Llama-3.2-1B**: quick sweeps across the full grid.
- **Llama-3.2-3B**: promote the strongest 1B settings.
- **Llama-3.1-8B**: final confirmation for top variants to match the CoLA paper setup

All runs share: seq length 256, AdamW lr = 1e-4, rank-8 adapters, 5 epochs, identical evaluation suites (ARC, MMLU, HellaSwag, Belebele, FLORES) to keep comparisons fair.

### Language Coverage / Data Size

Use the tiered subsets already defined in `distributed_data_processor/language_subsets.py`:

| Tier | Source symbol | # Languages | Notes |
| --- | --- | --- | --- |
| S | `twenty_two_representatives_mediods` (`distributed_data_processor/language_subsets.py:22`) | 22 | diverse scripts |
| M | `ninty_five_representatives_mediods` (`distributed_data_processor/language_subsets.py:96`) | 95 | balanced coverage |
| L | `hundred_ninty_nine_representatives_mediods` (`distributed_data_processor/language_subsets.py:200`) | 199 | “full” setting |

Cap per-language token counts so each tier processes roughly the same number of tokens per epoch (e.g., S: 20k/lang, M: 10k/lang, L: 5k/lang).

## Adapter Variants

1. **LoRA-r8 baseline** – `use_hydralora_experts=False`, `lora_num=1`, or `num_A=num_B=1`.
2. **HydraLoRA-flat** – enable multiple `B` heads without routing (`lora_num∈{2,4}`) to isolate asymmetric A/B behavior (`peft/tuners/hydralora/layer.py:147`).
3. **HydraLoRA + routing** – `use_hydralora_experts=True`, `hydralora_num_experts=#languages`, `hydralora_top_k∈{1,2}`, provide `language_ids` to bias experts via `language_router_mode` (`peft/tuners/hydralora/layer.py:395`).
4. **CoLA-flat** – `use_cola_experts=False`, vary `(num_A,num_B)∈{(1,1),(1,3),(2,3)}` and `cola_strategy∈{fully,random_ab,random_ba,heuristic}` (`peft/tuners/cola/layer.py:730`).
5. **CoLA + routing** – `use_cola_experts=True`, `cola_num_experts=#languages`, `cola_top_k=1`, share `A` by family using `hierarchical_low_resource_clusters`/`hierarchical_four_families` to realize “A=family, B=language” (`distributed_data_processor/language_subsets.py:396`; `peft/tuners/cola/layer.py:217`).

## Language Prior Routing & Losses

Following MoE-LPR:

- Stage 1: routers run in `learned` mode with load balancing.
- Stage 2: freeze experts, train routers only on a replay buffer (<1% of Stage 1 tokens) with `language_router_mode="bias"` or `"hard"` and an auxiliary CE weighted by `language_prior_weight ∈ {0.1, 0.3}`.
- Evaluate with and without the LPR loss to quantify its effect on catastrophic forgetting.

Both Hydra and CoLA layers already cache router logits/targets for metrics (`peft/tuners/hydralora/layer.py:474`; `peft/tuners/cola/layer.py:630`), enabling LPR loss computation and logging (entropy, load CV, target hit rate).

## Run Matrix

Per language tier × model size (3 tiers → 36 runs per model → 108 total across 1B/3B/8B):

1. `LoRA`
2. `Hydra-flat (lora_num=3)`
3. `Hydra + LPR (use_hydralora_experts, top_k=1)`
4. `CoLA-flat (num_A=1,num_B=3, strategy=fully vs random_ba)`
5. `CoLA + LPR (use_cola_experts, num_A=1,num_B=3)`
6. `CoLA family-shared (family-specific A, per-language B)`

Each variant runs twice (LPR off/on) → up to 12 configs per tier. Start on 1B, retain top performers (router metrics + benchmark accuracy) for 3B, then only the best 2 variants per tier for 8B to manage compute.

## Configuration Notes

- **Initialization**: use PiSSA (`init_lora_weights="pissa"`) for all CoLA runs to maintain stability at low sample counts (`papers/further/notes.md:24`; `peft/tuners/cola/layer.py:336`). HydraLoRA can keep the default Kaiming initialization (`peft/tuners/hydralora/layer.py:205`).
- **Routing Bias**: `language_bias_value=5.0` for `bias` mode (default in config) to push tokens toward their prior expert; adjust based on router entropy logs.
- **Evaluation**: Benchmark on FLORES and Belebele subsets (language lists already curated at `distributed_data_processor/language_subsets.py:452` and `distributed_data_processor/language_subsets.py:522`) plus ARC/MMLU/HellaSwag to match MoE-LPR and CoLA comparisons
- **Logging**: Persist router metrics via `record_hydralora_metrics`/`record_cola_metrics` hooks to analyze load balancing, entropy, and target hit rates (`peft/tuners/hydralora/layer.py:547`; `peft/tuners/cola/layer.py:809`).

## Alignment vs. Papers

- **CoLA**: We mirror the best-performing settings (rank-8, PiSSA init, #A < #B such as 1×3 and 2×3) reported for Llama-3.1/3.2 models. Our additions are the family-shared `A` blocks and explicit LPR stage, which the paper did not test.
- **HydraLoRA**: Paper configuration used r=8 with one shared `A` and multiple `B` heads (3 for single-domain, 10 for BBH) plus a router over intrinsic components. We keep the same rank/asymmetry but add the MoE-LPR-style replay stage and language-driven routing.
- **MoE-LPR influence**: The two-stage router review and LPR loss (Stage 2, replay <1% data) follow the original paper. Our divergence is adapting LoRA experts (Hydra/CoLA) instead of full FFNs and targeting Llama-3 backbones instead of Qwen1.5.
