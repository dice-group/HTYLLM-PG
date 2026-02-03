# Training Plan: Llama‑3‑8B Multilingual Adapters

## Base Setup
- **Backbone**: `meta-llama/Llama-3.1-8B` (bf16) with LoRA-friendly settings from our existing scripts.
- **Tokenizer / embeddings**: three tiers, each with its own vocab extension + embedding init already materialized alongside the checkpoints:
  - **Tier‑12** (tight family clusters, e.g. 4 langs/expert).
  - **Tier‑72** (mid-scale clustering from `72_tier_language_groupings.json`).
  - **Tier‑200** (full spread, `200_tier_language_groupings.json`).
- **Data budget**: follow `docs/decide_token_budget.md` smoothing (α=0.3) so each tier consumes its prescribed token share per language.

## Experiment Knobs
- **Adapters**: LoRA baseline (`finetuning_type=lora`), HydraLoRA flat (`use_hydralora_experts=False`, `lora_num=3`), HydraLoRA experts (`use_hydralora_experts=True`), CoLA flat (`use_cola_experts=False`, `num_A=1,num_B=3`) and CoLA experts (`use_cola_experts=True`, shared‑A per cluster).
- **Language guidance**: `language_guidance_scope=all` for expert runs so both expert routing and B-head selection see metadata; baselines keep `scope=none`.
- **Router mode + LPR weight**: learned (`γ=0`) for baselines; bias (`γ=0.1`) for the Tier‑12/Tier‑72 expert runs. We’ll introduce hard-mode / replay once the router-only stage is implemented.
- **Metrics**: rely on `docs/training_metrics.md` (load CV, target hit-rate, LPR loss) and FLORES/Belebele evals after each tier to monitor routing health and accuracy.

## Initial Tier‑12 Launch Set
To de-risk the pipeline and secure early paper-ready numbers we start with five runs on the 12-language tier:

| Run | Adapter | Guidance | `language_router_mode` | `language_prior_weight` | Notes |
| --- | --- | --- | --- | --- | --- |
| T12-L0 | LoRA | `none` | learned | 0.0 | `finetuning_type=lora` (rank-8) |
| T12-L0-Base | LoRA (base tokenizer) | `none` | learned | 0.0 | Uses original `meta-llama/Llama-3.1-8B` weights + base tokenized data |
| T12-HF | HydraLoRA flat | `none` | learned | 0.0 | `lora_num=3`, `use_hydralora_experts=False` |
| T12-HE | HydraLoRA experts | `all` | **bias** | **0.1** | `use_hydralora_experts=True`, cluster-guided experts with Stage 1 LPR |
| T12-CF | CoLA flat | `none` | learned | 0.0 | `use_cola_experts=False`, `num_A=1,num_B=3` |
| T12-CE | CoLA experts | `all` | **bias** | **0.1** | `use_cola_experts=True`, shared-A per cluster with Stage 1 LPR |

These runs span LoRA vs asymmetric flat adapters vs hierarchical experts, with the expert variants immediately using soft metadata guidance (bias + γ=0.1) so we can measure LPR’s benefit without waiting for Stage 2. Once the router-only replay flow is in place we can resume from these checkpoints for hard-mode ablations.

## Initial Tier‑72 Launch Set
As soon as the Tier‑12 configs prove stable we launch the same five variants on the 72-language tier (with the larger tokenizer/embedding and dataset sharding):

| Run | Adapter | Guidance | `language_router_mode` | `language_prior_weight` | Notes |
| --- | --- | --- | --- | --- | --- |
| T72-L0 | LoRA | `none` | learned | 0.0 | Same rank-8 LoRA baseline, promoted hyperparams |
| T72-HF | HydraLoRA flat | `none` | learned | 0.0 | Mirror T12-HF (`lora_num=3`, no experts) |
| T72-HE | HydraLoRA experts | `all` | **bias** | **0.1** | Full guidance via `use_hydralora_experts=True` + tier-72 JSON |
| T72-CF | CoLA flat | `none` | learned | 0.0 | Same collaborative A/B setup as Tier‑12 |
| T72-CE | CoLA experts | `all` | **bias** | **0.1** | Shared-A per cluster with immediate LPR |

This gives us a 10-run starting grid (5 per tier) that hits all major axes (LoRA vs flat vs expert with soft guidance) and lets us compare routing/accuracy trends across tiers before scaling further.
