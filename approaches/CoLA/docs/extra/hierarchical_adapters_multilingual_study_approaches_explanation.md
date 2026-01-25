# Ground Truth: Multilingual CoLA/HydraLoRA Routing Behaviors (ACL Runs)

This note defines the expected *stage-by-stage* behavior for the CoLA/HydraLoRA variants we plan to evaluate, so others can review configurations and confirm the code matches intent.

## Shared Concepts

- **Token language id**: each sequence in a batch may carry a `language_ids` value (shape `[B]` after flattening). `-1` means “unknown/pad”.
- **Experts (stage 1)**: a router selects one or more experts per token (`top_k`). Expert selection is *token-level* (per time step) because routing logits are `[B, S, E]`.
- **Heads / B-matrices (stage 2)**: within an expert, there can be multiple B heads (e.g., `num_B=3`). Head selection/weighting is applied *within that expert*.
- **Guidance / supervision (LPR)**: “LPR loss” (language-prior routing) supervises routers toward a language-derived target:
  - expert router targets: `language_id_to_expert`
  - head router/targets: `language_id_to_subgroup`
- **Mappings** (from clustering JSONs):
  - `language_list`: ordered list of languages (defines language index space).
  - `language_to_family_ids`: maps language index → expert id (tier/group).
  - `language_to_subgroup_ids`: maps language index → subgroup/head id (within expert).

## Configuration Knobs (Semantics)

- `language_guidance_scope`:
  - `none`: no language targets are computed/cached; no language bias/forcing.
  - `expert_only`: language targets apply to expert routing only (where supported).
  - `all`: language targets apply to expert routing and (where implemented) head routing/weighting.
- `language_router_mode`:
  - `learned`: router runs purely from router network outputs; targets may still be cached for LPR loss.
  - `bias`: add a constant logit bias `language_bias_value` to the target index (soft guidance).
  - `hard`: force selection to the target index (hard routing), when targets exist.
- `language_head_router_mode` (override; optional):
  - If set, applies to the *head/B stage only* (HydraLoRA head router, CoLA head emphasis).
  - If unset, defaults to `language_router_mode`.
- `language_bias_value`: strength of the `bias` mode. For head weighting with `H` heads:
  - target weight `w_t = exp(b) / (exp(b) + (H-1))`
  - non-target weight `w_o = 1 / (exp(b) + (H-1))`
- `language_head_bias_value` (override; optional): if set, applies to `bias` at the head/B stage only; defaults to `language_bias_value`.

## Approaches (Ground Truth)

## Preset Configs We Intend To Run (Reviewer Checklist)

These are the *intended* ACL paper cells, with explicit knobs (mirrors `README.md` / `scripts/comparison/run_multilingual_ablation.sh`).

**Soft LPR (loss-only)** (used for most “LPR=0.1” runs):
- `LANGUAGE_ROUTER_MODE=learned`
- `LANGUAGE_PRIOR_WEIGHT=0.1`
- `LANGUAGE_BIAS_VALUE=0.0`

**Soft bias guidance** (optional extra ablation; also used to create CoLA head emphasis Variant A):
- `LANGUAGE_ROUTER_MODE=bias`
- `LANGUAGE_BIAS_VALUE=2.0` (typical)
- optionally keep `LANGUAGE_PRIOR_WEIGHT=0.1`

### A0: LoRA baseline (adapter-only, no routing)

- `finetuning_type=lora`, standard LoRA config (no Hydra wrapper)
- `LANGUAGE_GUIDANCE_SCOPE=none`, `LANGUAGE_PRIOR_WEIGHT=0.0`

### A0: LoRA (baseline)

- Stage 1 (expert): none
- Stage 2 (head): none
- Expected behavior: one low-rank adapter per layer.

### C0: CoLA flat (paper-faithful)

- `finetuning_type=cola`, `USE_COLA_EXPERTS=False`, `NUM_A=1`, `NUM_B=3`, `COLA_STRATEGY=fully`
- `LANGUAGE_GUIDANCE_SCOPE=none`, `LANGUAGE_PRIOR_WEIGHT=0.0`

### C0: CoLA Flat (paper-faithful)

- No experts (`use_cola_experts=false`).
- Within the adapter, CoLA uses multiple A and B matrices with `cola_strategy`:
  - `fully`: sum over all A×B combinations.
  - `random_ab`, `random_ba`, `heuristic`: alternative collaboration patterns.
- Language guidance: not used (no expert router).

### C1: CoLA Experts (expert routing only; no head emphasis)

- `finetuning_type=cola`, `USE_COLA_EXPERTS=True`, `COLA_TOP_K=1`, `NUM_A=1`, `NUM_B=3`, `COLA_STRATEGY=fully`
- **Stage 1 (expert router, soft LPR)**: `LANGUAGE_GUIDANCE_SCOPE=all`, `LANGUAGE_ROUTER_MODE=learned`, `LANGUAGE_PRIOR_WEIGHT=0.1`
- **Stage 2 (B heads)**: collaborative / uniform inside each expert (no head router, no head emphasis)

- Experts enabled (`use_cola_experts=true`), router selects experts via top-k.
- Stage 1 (expert): router outputs `logits=[B,S,E]`, selects `top_k` experts and weights.
  - If `language_guidance_scope != none`, expert targets are cached for optional LPR loss.
  - If `language_router_mode=bias`, adds logit bias to the language-target expert.
  - If `language_router_mode=hard`, forces top-1 expert to the language-target expert.
- Stage 2 (head): fully collaborative inside each expert (no language-specific head weighting).
- Expected behavior: token may mix multiple experts (if `top_k>1`).

### C2: CoLA Experts + Head Emphasis (Variant A; current)

This keeps CoLA “fully collaborative” but *reweights* B-head contributions inside the target expert.

- `finetuning_type=cola`, `USE_COLA_EXPERTS=True`, `COLA_TOP_K=1`, `NUM_A=1`, `NUM_B=3`, `COLA_STRATEGY=fully`
- **Stage 1 (expert router, soft LPR)**: `LANGUAGE_GUIDANCE_SCOPE=all`, `LANGUAGE_ROUTER_MODE=learned`, `LANGUAGE_PRIOR_WEIGHT=0.1`
- **Stage 2 (B emphasis inside target expert)**:
  - soft: `LANGUAGE_HEAD_ROUTER_MODE=bias`, `LANGUAGE_HEAD_BIAS_VALUE=2.0` (can keep `LANGUAGE_PRIOR_WEIGHT=0.1`)
  - hard: `LANGUAGE_HEAD_ROUTER_MODE=hard`

- Stage 1 (expert): same as C1.
- Stage 2 (head): **only inside the language-target expert**, compute head weights `w_B` from `language_to_subgroup_ids`:
  - `language_head_router_mode=bias`: `w_B = softmax(bias on target head)`
  - `language_head_router_mode=hard`: one-hot on target head
  - otherwise: uniform
- If a token is routed to a *non-target* expert:
  - head target is masked out → head weights fall back to uniform (no language-specific B emphasis).
- Expected behavior:
  - When routed correctly, the expert’s output is still a sum of all B heads, but biased toward the language’s head.
  - When routed incorrectly, the expert behaves like fully-collaborative CoLA (uniform across heads).

### H0: HydraLoRA Flat (paper-faithful)

- `finetuning_type=hydralora`, `USE_HYDRALORA_EXPERTS=False`, `LORA_NUM=3`
- **Paper-faithful (no language supervision)**: `LANGUAGE_GUIDANCE_SCOPE=none`, `LANGUAGE_ROUTER_MODE=learned`, `LANGUAGE_PRIOR_WEIGHT=0.0`

- No experts (`use_hydralora_experts=false`).
- Stage 1 (expert): none.
- Stage 2 (head): **head router** mixes B heads:
  - router `lora_route` outputs per-token logits over heads, produces `route_weight=[B,S,H]`.
  - If `language_guidance_scope=all`, head targets are cached for optional LPR loss.
  - If `language_router_mode=bias`, add logit bias to the language-target head.
  - If `language_router_mode=hard`, force one-hot head selection.

### H1: HydraLoRA Experts (2-stage expert→head routing; current)

- `finetuning_type=hydralora`, `USE_HYDRALORA_EXPERTS=True`, `HYDRALORA_TOP_K=1`, `LORA_NUM=3`
- **Stage 1 (expert router, soft LPR)**: `LANGUAGE_GUIDANCE_SCOPE=all`, `LANGUAGE_ROUTER_MODE=learned`, `LANGUAGE_PRIOR_WEIGHT=0.1`
- **Stage 2 (head router within target expert, soft LPR)**: `LANGUAGE_GUIDANCE_SCOPE=all`, `LANGUAGE_ROUTER_MODE=learned`, `LANGUAGE_PRIOR_WEIGHT=0.1`

- Stage 1 (expert): expert router selects experts (`logits=[B,S,E]`, `top_k`).
  - If `language_guidance_scope in {all, expert_only}`, expert targets cached for LPR loss.
  - Bias/hard guidance apply the same way as CoLA expert routing.
- Stage 2 (head): within *each expert*, the Hydra head router mixes that expert’s B heads.
  - If `language_guidance_scope=all`, head targets are computed and cached for LPR loss.
  - Head targets are masked to only apply when the token’s language-target expert matches the current expert.
  - Bias/hard guidance apply at the head level (within that expert).
- Expected behavior:
  - Correct routing: token goes to target expert; inside it, head router prefers the target head (if configured).
  - Incorrect routing: token’s selected expert is not the target; head guidance is masked → head routing becomes unguided inside that expert.
- With `top_k>1`, both target and non-target experts can contribute, weighted by the expert router.

### C1b: CoLA Experts (expert-only guidance)

Same as C1, but restrict guidance to the expert router only.

- `finetuning_type=cola`, `USE_COLA_EXPERTS=True`, `COLA_TOP_K=1`, `NUM_A=1`, `NUM_B=3`, `COLA_STRATEGY=fully`
- **Stage 1 (expert router, soft LPR)**: `LANGUAGE_GUIDANCE_SCOPE=expert_only`, `LANGUAGE_ROUTER_MODE=learned`, `LANGUAGE_PRIOR_WEIGHT=0.1`
- **Stage 2 (B heads)**: no language guidance (head selection/weighting stays unguided)

### H1b: HydraLoRA Experts (expert-only guidance)

Same as H1, but restrict guidance to the expert router only.

- `finetuning_type=hydralora`, `USE_HYDRALORA_EXPERTS=True`, `HYDRALORA_TOP_K=1`, `LORA_NUM=3`
- **Stage 1 (expert router, soft LPR)**: `LANGUAGE_GUIDANCE_SCOPE=expert_only`, `LANGUAGE_ROUTER_MODE=learned`, `LANGUAGE_PRIOR_WEIGHT=0.1`
- **Stage 2 (head router)**: no language guidance (head routing stays learned-only)

## “What if language ids are missing?”

If `language_ids` are `None` or all `-1`, then:

- Expert routing runs without language targets (no bias/hard enforcement).
- Head routing/weighting runs without language targets (uniform or learned-only behavior).
- LPR loss has no usable targets for those tokens.

## Debug/Verification Hooks

- CoLA expert creation prints (when `cola_debug=true`): counts of experts and per-expert A/B sizes, plus a small language preview.
- CoLA routing sample prints: gated by env `COLA_DEBUG_ROUTING_EVERY=N` (prints every N-th forward call).
- HydraLoRA expert creation prints (when `hydralora_debug=true`): per-expert head counts, plus a small language preview.
- HydraLoRA routing sample prints:
  - if env `HYDRA_DEBUG_ROUTING_EVERY<=0`: prints every forward call
  - else prints every N-th forward call
