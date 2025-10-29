# Harmonized PEFT Architecture in the CoLA Repository

This document summarizes how the CoLA repository hosts a large collection of parameter-efficient fine-tuning (PEFT) approaches in a single, consistent code base. It highlights the abstractions that let very different adapters coexist, the way they are mapped into the training workflow, and practical takeaways you can reuse when building a MoE-style study that needs to compare many adapters side-by-side.

## 1. High-Level Architecture

- **PEFT Fork as Canonical Backend**  
  The `peft` directory is a mostly up-to-date mirror of Hugging Face’s PEFT library, extended with the CoLA adapter. Each adapter is implemented as a *tuner* (e.g., `peft/tuners/lora`, `peft/tuners/cola`) that subclasses the shared `BaseTuner` class (`peft/tuners/tuners_utils.py:79`).  
- **Registry-Based Discovery**  
  Central maps declare every available PEFT type: configuration, model wrapper, and runtime wiring all funnel through `peft/mapping.py:88`. Adding a new adapter only requires dropping a `Config`/`Model` pair under `peft/tuners/` and registering it in the dictionaries.
- **Unified Training Surface**  
  The `LLaMA-Factory` submodule provides the CLIs/Gradio apps. Once a `finetuning_type` is chosen, the training pipeline always builds a PEFT config, obtains a `PeftModel` via the registry, and hands it to the same trainer logic (`LLaMA-Factory/src/llamafactory/model/adapter.py:256` onwards). This makes “full”, “freeze”, LoRA variants, and prompt-based tuners plug-and-play.

## 2. Shared Abstractions Inside `peft/`

### 2.1 Configuration Contracts

- All adapters inherit `PeftConfig` (`peft/config.py`) and declare their hyperparameters and metadata (e.g., `peft/tuners/cola/config.py:24`, `peft/tuners/vera/config.py:24`).  
- The base class sets `peft_type`, `task_type`, and flags like `is_prompt_learning`, so downstream utilities can treat configs uniformly.
- Shared helpers (target module regexes, rank/alpha patterns, layer replication) live on the base config; individual tuners opt-in by reusing them.

### 2.2 Injection Pipeline

1. **User → Config** – `get_peft_config` converts raw JSON/dicts into typed configs (`peft/mapping.py:130`).  
2. **Config → Adapter Model** – `get_peft_model` wraps a `transformers.PreTrainedModel` with the requested tuner (`peft/mapping.py:141`). It uses registries to pick the correct `BaseTuner` subclass or one of the prompt-learning modules.  
3. **BaseTuner** – Handles adapter injection and bookkeeping (`peft/tuners/tuners_utils.py:79`). It:
   - Locates target modules (`_prepare_adapter_config`).  
   - Replaces them with adapter-aware layers (`_create_and_replace`).  
   - Marks gradient flags via `_mark_only_adapters_as_trainable`.  
   - Exposes hooks to merge/unmerge adapters, toggle active adapters, etc.
4. **BaseTunerLayer** – A shared wrapper around each modified layer (`peft/tuners/tuners_utils.py:223`). It unifies saving/loading, merges, scaling, and mixed precision for all adapter types.
5. **PeftModel** – A thin orchestrator that wraps the adapted model, handles saving, forwards, and optional prompt learning extras (`peft/peft_model.py:37`). From the trainer’s perspective, a `PeftModel` behaves like any `torch.nn.Module`.

This stack is why LoRA, HydraLoRA, CoLA, IA3, OFT, FourierFT, VeRA, etc. “just work” the same way once their configs are registered.

### 2.3 Adapter Diversity via Specialization Hooks

- **Override Points** – Tuners like CoLA and HydraLoRA override `BaseTuner._create_and_replace` to instantiate custom layer classes (`peft/tuners/cola/model.py:118`).  
- **Custom Layer Logic** – Layers provide their bespoke parameter creation and forward pass logic while inheriting merge/save utilities. For example, CoLA’s `ColaLayer` creates multiple A/B experts and PiSSA-initializes them (`peft/tuners/cola/layer.py:90`), while standard LoRA uses single A/B matrices with optional DoRA gating (`peft/tuners/lora/layer.py:101`).  
- **Shared Utilities** – Even with custom layers, shared helpers (e.g., `check_target_module_exists`, `replicate_layers`, and `dispatch_default`) keep the tuning logic decoupled from model-specific quirks.

## 3. Integration with the LLaMA-Factory Training Workflow

- **Argument Parsing** – `FinetuningArguments` defines the accepted `finetuning_type` values (`LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py:351`). Each entry toggles specific CLI options (e.g., `num_A`, `num_B` shown in the README recipes).  
- **Adapter Setup** – `_setup_lora_tuning`, `_setup_cola_tuning`, etc. map training arguments to PEFT config keywords and call `get_peft_model` (`LLaMA-Factory/src/llamafactory/model/adapter.py:266` and `:325`). The code paths are intentionally parallel, differing only in adapter-specific kwargs.  
- **Downstream Trainer** – Once the `PeftModel` is returned, the rest of LLaMA-Factory (data loading, optimizer construction, logging, evaluation) is agnostic to which adapter is attached. This ensures fair comparisons as long as hyperparameters are aligned.  
- **Adapter Persistence** – Training scripts reuse PEFT save/load utilities. Mixing and merging adapters is handled via `PeftModel.from_pretrained` calls in the same adapter setup functions (`LLaMA-Factory/src/llamafactory/model/adapter.py:288` onwards), so checkpoints produced by any tuner share the same format.

## 4. How Different Tuners Coexist Cleanly

- **Consistent Module Contracts** – All tuners operate on the same conceptual hooks: they receive the base layer, return an augmented layer exposing `enable_adapters`, `merge`, `set_scale`, etc. Training code never needs to branch once the adapter is injected.  
- **Shared Runtime Behaviors** – Features like mixed-precision casting, offloading, gradient checkpointing, and DeepSpeed integration are implemented once in `BaseTuner`/`PeftModel`. Every tuner inherits them, reducing divergence.  
- **Optional Specialized Behavior** – When an adapter needs something unusual (e.g., CoLA’s multi-expert initialization, FourierFT’s frequency table), that logic lives entirely inside its layer/config with no leakage into the trainer.  
- **Registries and Enum Guards** – `PeftType` (`peft/utils/peft_types.py:22`) enumerates every adapter. If the trainer sees an unknown `finetuning_type`, it fails early, preventing silent fallbacks to incorrect adapters.  
- **Consistent Serialization** – All adapters respect PEFT’s `save_pretrained` / `from_pretrained` contract, which serializes only the adapter parameters plus optional `modules_to_save`. This normalizes checkpoints for later comparison or stacking.

## 5. Lessons for a MoE Adapter Study

1. **Adopt a Common Injection Interface** – Model your adapters after `BaseTuner` / `BaseTunerLayer`. Even if MoE adapters are structurally different (routing tables, expert banks), enforce a shared `update_layer`, `set_adapter`, and `merge` API so the training loop stays identical.  
2. **Centralize Registry Metadata** – Maintain a single source of truth (enums + dictionaries) that map adapter identifiers to configs/models. It simplifies CLI integration and reduces boilerplate when adding new methods.  
3. **Keep Specialized Logic Encapsulated** – Let MoE-specific behavior live inside the adapter’s layer or config. Avoid branching in the training pipeline; it should only translate arguments into config kwargs.  
4. **Unify Persistence** – Standardize how checkpoints store adapters. If you support stacking/merging MoE experts, piggyback on `save_pretrained`-style utilities so every approach can be resumed with the same code.  
5. **Expose Comparable Hyperparameters** – As CoLA does with `num_A`/`num_B`, surface knobs that make parameter budgets explicit. It makes benchmarking easier and encourages fair comparisons.  
6. **Leverage Shared Utilities** – Generic helpers for discovering target modules, replicating layers, or handling quantized weights remove repeated code paths and reduce the chance of discrepancies across adapters.

## 6. Extending the Pattern

If you plan to integrate additional adapters or MoE methods:

- Add a new folder under `peft/tuners/<name>` with `config.py`, `model.py`, and optionally `layer.py`. Inherit from `BaseTuner`/`BaseTunerLayer`.  
- Register the config and model in `peft/mapping.py` and add the type to `PeftType`.  
- Expose CLI arguments in `FinetuningArguments` if you want first-class command-line support.  
- Reuse `_setup_<adapter>_tuning` as a template to translate CLI flags into `get_peft_model` kwargs.

By following these patterns, you can grow a MoE research repository while keeping adapters interoperable, comparable, and easy to train through a unified workflow—exactly the strengths demonstrated in CoLA’s PEFT integration.

