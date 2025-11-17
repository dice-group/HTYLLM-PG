# CoLA Language-Specific Routing Notes

## Use Case & Goal
- Target: massively multilingual fine-tuning where data arrives grouped by language.
- Observation: in practice A matrices tend to capture family-level structure, while B matrices specialize to individual languages.
- Desired change: remove the learned per-token router for CoLA experts and instead route every batch deterministically based on the language family. Each family keeps one shared A stack and exposes multiple B matrices, one per language in that family.

## Current Implementations
- **CoLA (`peft/tuners/cola/layer.py`)**
  - `use_cola_experts` enables a standard MoE router (linear layer + top-k softmax) that scores every expert directly from token representations.
  - Each expert currently contains `num_A` A matrices and `num_B` B matrices; the forward loop exhaustively combines them (fully collaborative strategy).
  - PiSSA init supports either per-adapter or shared initialization when experts are active.

- **HydraLoRA (`peft/tuners/hydralora/layer.py`)**
  - Non-MoE mode already shares a single A with multiple B heads per adapter and uses a learned per-token router (`lora_route`) to weight those B heads.
  - When `use_hydralora_experts` is set, Hydra mirrors CoLA’s MoE behavior: a single router selects experts, and `_adapter_delta` sums that expert’s B heads.
  - There is no explicit two-stage routing that first picks an A/family and then a language-specific B.

## Planned Direction
High level:
1. Deterministic, metadata-driven routing for language families (optional learned fallback when metadata absent).
2. Shared A stack per family; multiple B stacks per language.
3. Training/inference APIs accept `language_id` to bias or override routing.

### Implementation TODOs
1. **Config plumbing**
   - Extend `ColaConfig` / Hydra config to carry language + family metadata and routing-mode flags.
   - Update `FinetuningArguments` and PEFT model creation so those fields reach the tuners.
2. **Router inputs**
   - Modify `ColaLayer.forward` (`peft/tuners/cola/layer.py:607+`) to accept optional `language_id`, expose router logits, and allow hard/biased routing when metadata is present.
   - Apply the same pattern to Hydra routing (`peft/tuners/hydralora/layer.py:330-433`).
3. **Parameter sharing**
   - Refactor `ColaLayer.update_layer` so experts reference family-level A tensors rather than duplicating them; add mapping tables (family→A modules, language→B modules).
   - Mirror the mapping logic for Hydra’s `lora_A`/`lora_B` so families and languages are first-class indices.
4. **Language-prior loss hook**
   - Provide a mechanism (e.g., cached logits) for the trainer to compute an auxiliary loss using `language_id`.
   - Thread a new hyperparameter (γ) through training args to weight this loss.
5. **Initialization**
   - Adjust PiSSA/shared init paths to seed each family A stack once, then clone for languages, preserving the current dtype/FSDP safeguards.
6. **Docs & tests**
   - Update CLI/help text for new metadata flags and routing behaviors.
   - Add regression tests or scripted checks covering metadata-driven routing vs learned fallback.

This document tracks the actionable plan, while `docs/cola_language_prior_routing.md` remains the broader conceptual reference.
