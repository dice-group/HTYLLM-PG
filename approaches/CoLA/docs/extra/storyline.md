# Hierarchical Language-Structured Adapters with Language Prior Routing

This README describes the core idea, experimental story, and implementation plan for a paper on hierarchical multilingual adapters, including what baselines and infrastructure are needed.

---

## 1. Core Idea

We build **Language Prior Routing (LPR)** on top of existing CoLA/Hydra adapters:

1. **Hierarchical experts**: each Transformer layer hosts multiple adapter experts (`peft/tuners/cola/layer.py` MoE router or Hydra’s router). The router can be metadata-driven or purely learned.
2. **Family-shared A, language-specific B**: per expert we share A stacks across a language family and keep multiple B stacks for individual languages (`num_A`, `num_B` knobs already exist in CoLA/Hydra).
3. **Language prior loss**: batches carry `language_id`; routers expose logits so the trainer can add an auxiliary CE loss that nudges routing toward the language’s designated expert/head while still allowing fallback to hidden-state-only routing at inference.

This combination targets multilingual interference by enforcing structure while staying compatible with today’s CoLA/Hydra code paths.

---

## 2. Architecture Sketch

### 2.1 Layer Layout

- **Experts per layer**: reuse the existing CoLA MoE path (`use_cola_experts`) or Hydra expert mode; router = `nn.Linear` + top-k (`peft/tuners/cola/layer.py:625-676`).
- **Intra-expert structure**: retain CoLA’s multi-`A`/multi-`B` collaborative loops and Hydra’s shared-A multi-B heads (`peft/tuners/hydralora/layer.py:330-387`). Families map to shared A tensors, languages to B stacks.
- **Language gating**: forward paths accept optional `language_id`; if present we override or bias the router output and record logits for the LPR auxiliary loss. Without metadata the existing behavior (hidden-state routing) is unchanged.

### 2.2 Loss & Inference

- **Training loss**: `L_total = L_task + α·L_balance (optional) + γ·L_lang`, where `L_lang` is CE over the router logits targeting the language’s designated expert/head.
- **Inference modes**:
  1. Learned (no metadata, default).
  2. Metadata-hard (one-hot gating by `language_id`).
  3. Metadata-bias (additive logit bias toward the language’s expert).

---

## 3. Experimental Story

- **Model scales**: run 1B (local) and 8B (cluster) checkpoints via `accelerate_moe_cola_train.sh` / `train_moe_cola.sh`.
- **Language regimes**: 5-language smoke tests, 50-language mid-scale, 200-language full-scale; each dataset tagged with `language_id`.
- **Benchmarks**: XNLI, FLORES, internal multi-task mixtures; each evaluation scripted through LLaMA-Factory’s eval hooks or `lm_eval_checkpoint.sh`.
- **Hypothesis**: LPR adapters should retain quality as #languages grows, compared with flat LoRA/CoLA/Hydra baselines.

---

## 4. Baselines & Ablations

**Baselines (implemented via existing scripts/configs):**
1. LoRA (single adapter) – `--finetuning_type lora`.
2. HydraLoRA (shared A, multi-B, no experts) – `--finetuning_type hydralora`.
3. CoLA without MoE – `--finetuning_type cola`, `--use_cola_experts False`.
4. CoLA with MoE but no LPR – `--use_cola_experts True`, current router.
5. Flat MoE (multiple adapters, no hierarchical sharing) – configure multiple adapters without family mapping.

**Ablations to run via grid scripts (`grid_search_cola.sh` etc.):**
- Experts per layer (K ∈ {1,2,4}).
- `num_A`, `num_B` combos (1×1 vs 2×4).
- LPR off vs on (γ = 0 vs γ > 0).
- Metadata modes at inference (learned vs hard vs bias).
- Family clustering strategies.

---

## 5. Implementation Plan

- **Adapter changes**: follow `docs/cola_language_plan.md` TODOs—config plumbing, shared A/B maps, router hooks, PiSSA replication.
- **Language metadata**: extend dataloaders to pass `language_id`; LLaMA-Factory CLI args expose new flags (`--language_map`, `--lpr_weight`, etc.).
- **Training scripts**: update `accelerate_moe_cola_train.sh`, `train_moe_cola.sh`, and Slurm launchers to accept the new flags, propagate to `llamafactory-cli`, and log router stats.
- **Loss integration**: modify training loop (LLaMA-Factory) to compute `L_lang` using router logits cached by CoLA/Hydra layers.
- **Evaluation**: reuse `lm_eval_checkpoint.sh` for standardized scoring; add hooks to record per-language metrics.

---

## 6. HPC & End-to-End Pipeline

- **Scheduler**: Slurm scripts (`accelerate_moe_cola_train.sh`, `hydralora_slurm.sh`, etc.) already manage env setup, accelerate configs, WANDB logging, and optional checkpoint listeners (`checkpoint_listener.sh`). Extend them with language metadata flags and evaluation hooks.
- **Multi-node execution**: rely on Accelerate configs (`LLaMA-Factory/examples/accelerate/*.yaml`) or FSDP; ensure router state is BF16-friendly (see `training_errors.md` notes).
- **Automation**:
  - Use `grid_search_cola.sh` / `launch_accelerate_moe_cola_pair.sh` templates to sweep (model, languages, adapter, routing mode). Each config → Slurm job.
  - After training, call `lm_eval_checkpoint.sh` or a custom evaluator to score checkpoints; aggregate via simple Python scripts (e.g., under `tool/`).
- **Artifacts**:
  - Logs: WANDB runs + router statistics.
  - Checkpoints: saved via LLaMA-Factory’s `get_peft_model_state_dict`.
  - Reports: tables/plots comparing baselines vs LPR for each language regime.

---
