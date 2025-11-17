
# CoLA Language-Specific Routing with Language Prior Routing (LPR)

## Use Case & Goal

- Target: massively multilingual fine-tuning where data arrives grouped by language.
- Empirical observation:
  - A matrices tend to capture **family-level** structure.
  - B matrices specialize to **individual languages**.
- Goal:
  - Explicitly structure adapters by language family and language.
  - Use **Language Prior Routing (LPR)** so routing can:
    - exploit language metadata during training, and
    - work **without** metadata at inference (purely from hidden states).

---

## Current Implementations (Context)

### CoLA (`peft/tuners/cola/layer.py`)

- `use_cola_experts` enables a standard MoE router:
  - linear layer + top-k softmax over experts, fed by token representations.
- Each expert currently contains:
  - `num_A` A matrices and
  - `num_B` B matrices.
- Forward uses a fully collaborative strategy (all A with all B).
- PiSSA init supports:
  - per-adapter initialization,
  - shared initialization when experts are active.

### HydraLoRA (`peft/tuners/hydralora/layer.py`)

- Non-MoE mode:
  - One shared A per adapter.
  - Multiple B heads.
  - Learned per-token router `lora_route` weights B heads.
- Expert mode (`use_hydralora_experts`):
  - Global MoE router over experts, mirroring CoLA.
  - `_adapter_delta` sums that expert’s B heads.
- No explicit notion of “family-level A + language-specific B” yet.

---

## Design Overview: Language Prior Routing

We keep the existing learned routers (CoLA MoE router, Hydra `lora_route`), but:

1. **Structure parameters by (family, language):**
   - Families share A stacks.
   - Languages within a family have language-specific B stacks.
2. **Train routers with a Language Prior loss:**
   - During training, batches have known `language_id`.
   - We define a target expert / head for that language.
   - We add an auxiliary cross-entropy loss that encourages the router to put probability mass on that target.
3. **Inference modes:**
   - **Learned**: router uses hidden states only (no metadata).
   - **Metadata-hard** (optional): force route to the expert/head for a given language.
   - **Metadata-bias** (optional): add a bias towards the language’s expert/head.

---

## Language & Family Metadata

Assume a closed set of languages for an experiment.

### Config fields (conceptual)

At adapter-config level (shared across CoLA and Hydra):

- `languages: List[LanguageId]`
  - e.g. `["en", "de", "fr", "es", ...]`
- `families: List[FamilyId]`
  - e.g. `["germanic", "romance", "slavic", ...]`
- `lang_to_family: Dict[LanguageId, FamilyId]`
- `family_to_langs: Dict[FamilyId, List[LanguageId]]`

Derived integer indices:

- `family_index: Dict[FamilyId, int]`
  - maps each family to an integer family index.
- `expert_index: Dict[LanguageId, int]` (for CoLA and Hydra expert mode)
  - unique expert index per language (implicitly `(family, language)`).
- `head_index: Dict[LanguageId, int]` (for Hydra non-MoE)
  - head index per language within an adapter.

### Batch-level assumption

- Each batch is **single language**, and dataloader provides:
  - `language_id` (scalar per batch).

---

## CoLA + Language Prior Routing

### Parameterization

We interpret CoLA experts as **(family, language)** experts:

- For each language `ℓ`:
  - `family = lang_to_family[ℓ]`
  - `expert_index[ℓ] = e` (0 ≤ e < E).
- For each expert `e`:
  - `family[e] = f`
  - `language[e] = ℓ`
  - `A_e := A_family[f]` (shared A stack for the family)
  - `B_e := B_{f,ℓ}` (language-specific B stack)

A-family structure:

- Start with **M = 1** A per family (simplest).
- Later extend to **M > 1** per family, using CoLA’s collaborative strategies internally (see below).

### Forward with Learned Router

We reuse CoLA’s MoE router, but now its expert dimension corresponds to `(family, language)`.

Token-level variant (closest to current code):

1. For each token representation `x_t`:
   - `logits_t = router(x_t)  # shape [E]`
   - `g_t = softmax(logits_t)`.
2. Adapter delta for token `t`:
   - `ΔW_t = Σ_e g_t[e] · (B_e @ A_e)`
   - possibly with top-k sparsity.

Sequence-level variant (optional optimization):

1. Pool hidden states to `x̄` (e.g. mean or [CLS]).
2. `logits = router(x̄)`, `g = softmax(logits)`.
3. Use the same `g` for all tokens in the batch.

### Language Prior Loss (Training)

For a batch with language `ℓ*`:

1. Compute target expert index:
   - `e* = expert_index[ℓ*]`.
2. Define Language Prior loss:

- Token-level gating:
  - `L_lang = - mean_t log g_t[e*]`.
- Sequence-level gating:
  - `L_lang = - log g[e*]`.

3. Total loss:
   - `L_total = L_task + α · L_balance + γ · L_lang`
     - `L_task`: standard training loss (e.g. LM cross-entropy).
     - `L_balance`: optional load-balancing / entropy for MoE.
     - `L_lang`: language-prior loss.
     - `α, γ`: hyperparameters.

Effect:

- Router is **encouraged** (not forced) to use the expert for `ℓ*`.
- It can still spread probability if beneficial; LPR just biases routing.

### Inference Modes

When running with CoLA experts:

1. **Learned (default)**:
   - No language metadata required.
   - Use router as usual:
     - `g = softmax(router(x))`
     - `ΔW_t = Σ_e g_t[e] · (B_e @ A_e)`.
2. **Metadata-hard** (optional):
   - If `language_id` is provided, override router:
     - `e* = expert_index[language_id]`.
     - Replace `g_t` with one-hot vectors at `e*`.
3. **Metadata-bias** (optional):
   - If `language_id` is provided, bias logits:
     - `e* = expert_index[language_id]`.
     - `logits_t[e*] += bias_value` before softmax.
     - Larger `bias_value` → behavior closer to Metadata-hard.

### Extension to M > 1 (True CoLA Within Families)

When `num_A > 1` per family:

- For family `f`:
  - `A_{f,1..M}`.
- For language `ℓ` in family `f`:
  - `B_{f,ℓ,1..N_ℓ}`.

Within each expert `(f, ℓ)`, we define an internal CoLA composition, for example:

- **Fully collaborative**:
  - `A_f_total = Σ_i A_{f,i}`
  - `B_fℓ_total = Σ_j B_{f,ℓ,j}`
  - `B_e @ A_e := B_fℓ_total @ A_f_total`.

Alternative CoLA strategies (e.g., heuristic splits) can be plugged in later, without changing the router or LPR.

---

## HydraLoRA + Language Prior Routing

We add LPR on top of existing Hydra routing, both in non-MoE and MoE modes.

### Non-MoE Hydra (Per-Family Adapter)

Per adapter (e.g. one adapter per family):

- Single shared `A_f`.
- B heads for languages in the family:
  - `B_{f,ℓ}`.

Existing Hydra forward:

1. `g = softmax(lora_route(x))  # weights over heads`
2. `ΔW = Σ_h g[h] · (B_h @ A_f)`.

To integrate language priors:

1. Map languages to heads:
   - `head_index[ℓ] = h*` for each language `ℓ` in the family.
2. For a batch with language `ℓ*`:
   - `h* = head_index[ℓ*]`.

Language Prior loss:

- Token-level:
  - `L_lang = - mean_t log g_t[h*]`.
- Sequence-level:
  - `L_lang = - log g[h*]`.

Total loss:

- `L_total = L_task + γ · L_lang`  
  (plus any existing Hydra regularizers).

Inference:

- **Without metadata**:
  - Use `g` as in vanilla Hydra; routing depends only on hidden states.
- **With metadata (optional)**:
  - Hard override: set `g` to one-hot at `h*`.
  - Bias: add a fixed logit bonus to `h*` before softmax.

### Hydra Expert Mode (Optional)

If `use_hydralora_experts` is enabled:

- Treat global experts similarly to CoLA:
  - `expert_index[ℓ] = e` mapping languages to experts.
- Hydra’s global router outputs `g_t[e]`.
- Attach the same LPR loss:
  - `L_lang = - mean_t log g_t[e*]` with `e* = expert_index[ℓ*]`.
- Inside each expert `e`, you can:
  - keep a single B (simpler), or
  - use additional Hydra-style multi-B routing locally if needed.

---

## Initialization & PiSSA Compatibility

PiSSA and existing init remain unchanged; we only change how initialized A/B are *replicated* across families and languages.

### M = 1 Per Family

- Run PiSSA once to obtain a base A/B template per layer.
- For each family `f`:
  - Initialize `A_f` from the shared A template (exact copy).
- For each language `ℓ` in family `f`:
  - Initialize `B_{f,ℓ}` from the shared B template (exact copy or with small language-specific noise).

### M > 1 Per Family

- Use CoLA’s PiSSA-based scheme to obtain multiple A components for each layer.
- For each family `f`:
  - Copy the set `{A_{layer,1..M}}` to all families.
- For each language `ℓ` in `f`:
  - Create `B_{f,ℓ,1..N_ℓ}` from a shared B template (again, optionally plus small noise).

No changes to the PiSSA code path are required; only the replication and indexing differ.

---

## API & Training Integration

### Forward Signature

Extend adapter forward (CoLA and Hydra) to accept an optional `language_id`:

- `language_id` is a scalar per batch (string or int, mapped via config).
- Usage:
  - During training:
    - needed to compute `L_lang`.
  - During inference:
    - optional:
      - if absent: router behaves purely learned.
      - if present: can enable metadata-hard or metadata-bias modes.

### Loss Integration

- Routers (CoLA MoE, Hydra `lora_route`) must expose:
  - either the probabilities `g` (after softmax),
  - or the logits (so `L_lang` can be computed externally).
- Training loop:
  1. Run forward pass, get router outputs and main loss.
  2. Compute `L_lang` using `language_id` and mapping (`expert_index` or `head_index`).
  3. Combine into `L_total` with hyperparameter `γ`.
  4. Backpropagate `L_total`.

---

## Summary

- **Structure**: A matrices are shared within language families; B matrices are language-specific.
- **Routing**: CoLA and Hydra keep their learned routers, but:
  - languages are mapped to specific experts/heads, and
  - routers are trained with a Language Prior loss to favor those.
- **Inference**: works without language metadata (pure learned routing), but can optionally exploit metadata via hard overrides or biased logits.
- **Compatibility**: PiSSA and existing CoLA/Hydra code paths are reused; changes are localized to:
  - parameter indexing,
  - router targets,
  - and an additional loss term.
