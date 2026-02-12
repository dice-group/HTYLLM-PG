# Model Training and Implementation

This section documents the core adapter implementations and how language-prior routing (LPR) is integrated end-to-end.

---

## 1) Core Adapter Implementations

We extended the original CoLA implementation (https://github.com/zyy-2001/CoLA/tree/master) and specifically tailored the CoLA and HydraLoRA tuners for massively multilingual use cases.
We added expert support for CoLA, extended PiSSA initialization, and added multi‑node/FSDP support for large‑scale training via Accelerate. We also added the LPR implementation and multiple routing modes (expert‑only routing, head‑bias routing, standard LPR routing).

Everything is configurable. The most important configurable pieces are described below.

Detailed explanations about the modifications and configurations can be found here:
`approaches/CoLA/docs/extra/hierarchical_adapters_multilingual_study_approaches_explanation.md`

### 1.1 CoLA
- **Code**: `LLaMA-Factory/src/peft/tuners/cola/`
- **Key files**:
  - `config.py` (adapter config + routing flags)
  - `layer.py` (forward logic, expert routing, subgroup heads)
  - `model.py`, `forward.py`

### 1.2 HydraLoRA
- **Code**: `LLaMA-Factory/src/peft/tuners/hydralora/`
- **Key files**:
  - `config.py` (adapter config + routing flags)
  - `layer.py` (shared-A, multi-B heads, expert routing)
  - `model.py`, `forward.py`

---

## 2) Language-Prior Routing (LPR)
### 2.1 Metadata Flow
- **Language IDs injected**: `LLaMA-Factory/src/llamafactory/data/processor/supervised.py`
- **Batched into tensors**: `LLaMA-Factory/src/llamafactory/data/collator.py`
- **Language map parsing**: `LLaMA-Factory/src/llamafactory/extras/language.py`

### 2.2 Routing Logic
- **Routing utilities**: `LLaMA-Factory/src/peft/tuners/utils/language_routing.py`
- **CoLA routing**: `LLaMA-Factory/src/peft/tuners/cola/layer.py`
- **Hydra routing**: `LLaMA-Factory/src/peft/tuners/hydralora/layer.py`

### 2.3 LPR Loss Integration
- **Trainer hook**: `LLaMA-Factory/src/llamafactory/train/sft/trainer.py`
- **Aux loss**: `language_prior_weight` scales CE loss over routing targets

---

## 3) Adapter Configuration Plumbing
- **Hparams**: `LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py`
- **Adapter setup / validation**: `LLaMA-Factory/src/llamafactory/model/adapter.py`
- **Language map usage**: CoLA/Hydra configs accept `language_map`, `language_router_mode`, `language_guidance_scope`, `language_prior_weight`.

---

## 4) Training Variants (Conceptual)
- **LoRA baseline**: no experts, no routing.
- **Hydra flat**: shared-A, multi-B, no expert routing.
- **Hydra expert**: expert router + optional subgroup heads.
- **CoLA flat**: collaborative A/B, no expert routing.
- **CoLA expert**: expert router + subgroup heads.

---

## 5) Metrics and Diagnostics
- **Reference**: `docs/training_metrics.md`
- Logged via `record_cola_metrics` / `record_hydralora_metrics` inside adapter layers and trainer.
