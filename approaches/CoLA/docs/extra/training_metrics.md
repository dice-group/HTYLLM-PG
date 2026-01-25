# Training Metrics & Diagnostics

This project logs routing + LPR diagnostics directly from the adapter layers so we can see whether the hierarchy behaves as intended while training across 12/72/200 language tiers. All metrics are emitted through `record_cola_metrics` / `record_hydralora_metrics` and end up in W&B via the trainer’s periodic `log(pop_tracked_metrics())`.

## CoLA Expert Routing
- **`cola/expert_load_cv`** – Coefficient of variation of expert token counts. Low values ⇒ balanced routing; spikes signal collapse to a few experts.
- **`cola/active_expert_frac`** – Fraction of experts receiving any tokens during the step. Watch for drops when scaling the number of languages.
- **`cola/router_entropy`** – Mean entropy of the softmax logits across experts. High entropy = uncertain router; low entropy indicates confident dispatching.
- **`cola/topk_weight_mean`** – Average softmax weight assigned to selected experts. Useful for spotting whether top‑k blending is dominated by a single head.
- **`cola/expert_load_max_frac`** / **`…_min_frac`** – Largest/smallest share of routed tokens per expert. Use to visualize imbalance in addition to CV.

## CoLA Language-Target Metrics
- **`cola/language_target_hit_rate`** – Share of valid tokens whose top‑1 expert matches the metadata target. This is the clearest proxy for LPR effectiveness.
- **`cola/language_target_prob_mean`** – Mean probability assigned to the metadata expert. Rising curves show routers aligning even if top‑1 occasionally differs.
- **`cola/language_target_neglogp`** – Average `-log p(target)` (i.e., CE) before γ scaling. Use it to compare raw router alignment across γ values or tiers.
- **`cola/language_target_token_frac`** – Portion of tokens that actually had valid language metadata. Should stay near 1; drops indicate dataset issues.

## Hydra Expert Routing (MoE mode)
- **`hydralora/expert_load_cv`**, **`…_active_frac`**, **`…_router_entropy`**, **`…_topk_weight_mean`**, **`…_load_max_frac`**, **`…_load_min_frac`** – Same interpretations as the CoLA metrics but for Hydra’s expert router. They confirm whether the shared‑A experts stay balanced once you flip on `use_hydralora_experts`.
- **`hydralora/expert_target_*`** (hit rate / prob mean / neglogp / token frac) – Track alignment between LPR metadata and the expert router. Compare against CoLA to see which adapter makes better use of priors.

## Hydra Head Routing (flat Hydra mode)
- **`hydralora/head_load_cv`**, **`…_active_frac`**, **`…_router_entropy`**, **`…_load_max_frac`**, **`…_load_min_frac`** – Describe how evenly the per-adapter B heads are utilized. Plotting them per language tier reveals whether subclusters really split across different B matrices.
- **`hydralora/head_target_*`** – Same as the expert targets but for the intra-adapter head router. They are only populated when metadata is provided and tell you if subgroup guidance works.

## Language Prior Loss
- **`language_prior_loss_raw`** – Mean cross-entropy between router logits and metadata targets (no γ). It isolates router alignment dynamics independently of the chosen weight.
- **`language_prior_loss`** – Weighted auxiliary loss (`γ · raw`). Track it alongside the task loss to ensure the auxiliary term doesn’t dominate.

## How to Use These Metrics
1. **During training dashboards**: plot load CV/entropy/hit-rate for CoLA and Hydra side by side to detect expert collapse early. Add the raw/weighted LPR loss curves to confirm γ is set sensibly.
2. **Per-tier comparisons**: compare `expert_load_max_frac` across tiers (12→200 languages) to prove the hierarchy keeps experts balanced even as the routing space grows.
3. **Ablations**: when sweeping `language_router_mode` (“learned” vs “bias” vs “hard”), use the target hit-rate + neglogp metrics to quantify gains and tie them to downstream accuracy gaps.
4. **Paper plots**: export the W&B traces to show (a) routers remain balanced, (b) LPR accelerates alignment, and (c) subgroup heads are actually used. Pair these with per-language eval tables to make the reviewers’ lives easy.
