# Language-ID Aware lm-eval Wrapper (CoLA/Hydra)

This note captures only the information needed to implement a clean, minimal (KISS) wrapper so we can run **two evals**:
1) **No language IDs** (current behavior)
2) **With language IDs** (language‑prior guidance active)

It also documents exactly where language IDs are injected in the model and how to map `lm_eval` tasks to language IDs.

---

## Why a wrapper is needed
`lm_eval` does not pass `language_ids` into the model. In CoLA/Hydra, language‑prior routing (bias/hard/LPR targets) is **disabled** when `language_ids` are missing.

Code path (no IDs → no guidance):
- `language_expert_targets()` returns `None` if `language_ids` is `None`. (`LLaMA-Factory/src/peft/tuners/utils/language_routing.py`)
- `language_head_targets()` returns `None` if `language_ids` is `None`. (same file)
- CoLA headbias only applies when `head_targets` is not `None`. (`LLaMA-Factory/src/peft/tuners/cola/layer.py`)

Therefore, **lm_eval without IDs is not equivalent to training behavior**.

---

## How language IDs reach the model
The PEFT wrapper accepts `language_ids` and injects it via pre‑forward hooks:
- `PeftModel.special_peft_forward_args = {"adapter_names", "language_ids", "family_ids"}`
- `ColaModel._enable_peft_forward_hooks` registers forward pre‑hooks when `language_ids` is passed.
  (`LLaMA-Factory/src/peft/tuners/cola/model.py`)
- Same pattern for HydraLora (`LLaMA-Factory/src/peft/tuners/hydralora/model.py`)

**Implication:** The wrapper only needs to call the model with `language_ids=<tensor>`; no core model changes required.

---

## Language ID mapping
We must map `lm_eval` tasks to the internal `language_ids` used in training.

### Source of truth
- Adapter config (`adapter_config.json`) contains `language_list` in sorted order.
- This list aligns with the training `language_column` values.

### Task → language code
`configs/lm_eval_tasks.txt` uses task names like:
- `belebele_eng_Latn`
- `belebele_arb_Arab`
- `xnli` (no explicit language)

Recommended rule (KISS):
- If task name contains a language suffix after the first underscore, use that suffix as the language code.
  Example: `belebele_eng_Latn` → `eng_Latn`
- If no language suffix exists (e.g. `xnli`), **skip language‑aware eval** or run in no‑ID mode.

### Mapping to `language_ids`
```
lang_list = adapter_config["language_list"]  # list[str]
lang_id = lang_list.index(lang_code)
```
If not found, log and skip or fallback to no‑ID.

---

## Two evaluation modes

### 1) No language IDs (current)
Use existing script:
- `scripts/lm_eval_checkpoint.sh`

This provides the “metadata‑free” baseline.

### 2) Language‑ID aware eval
Implement a small wrapper that injects `language_ids` into every forward call.

Two KISS approaches:

#### Option A (Simplest, per‑task invocation)
Loop over tasks and run lm_eval **one task at a time**, passing a fixed `LANGUAGE_ID` for that task.
- No changes to lm_eval internals.
- Each invocation uses a custom model wrapper that reads a fixed ID from env.

#### Option B (Single run, custom LM class)
Implement a custom `LM` class that:
- Tracks the current task name.
- Maps task → language ID.
- Injects `language_ids` into every `model_call`.

Option A is easier and stable; Option B is more efficient.

---

## Minimal wrapper design (Option A)

### Files to add
- `scripts/lm_eval_language_ids.py` (per‑task wrapper with optional `--limit`)

### Inputs
- `--checkpoint` (adapter dir)
- `--tokenizer`
- `--tasks` (comma‑list or file)
- `--output-dir`
- `--limit` (optional, forwarded to lm_eval `simple_evaluate`)

### Steps
1) Load `adapter_config.json` to get `language_list`.
2) Parse tasks list into `(task_name, lang_code)`.
3) For each task with a valid `lang_code`:
   - `lang_id = language_list.index(lang_code)`
   - Set `LANGUAGE_ID=lang_id` env var
   - Call `lm_eval` for **that single task** using a custom model wrapper that reads `LANGUAGE_ID`.
4) For tasks without a language code, either:
   - skip, or
   - run in no‑ID mode and mark as such in outputs.

### Minimal model wrapper (pseudo)
```
class HFLMWithLang(HFLM):
    def __init__(..., language_id: int | None):
        self.language_id = language_id

    def _model_call(self, inputs, ...):
        if self.language_id is not None:
            lang_ids = torch.full((inputs["input_ids"].size(0),), self.language_id, device=inputs["input_ids"].device)
            return self.model(**inputs, language_ids=lang_ids)
        return self.model(**inputs)
```

If you don’t want to modify lm_eval internals, create a thin wrapper class and register it under a new model name (e.g. `hf_lang`) and call `lm_eval --model hf_lang`.

--- 

## Local PEFT note (CoLA/Hydra)
The wrapper force‑loads the repo-local `peft` package so CoLA/Hydra adapter types are available,
even if the site‑packages `peft` lacks them. This is required for `peft_type=COLA` and `HYDRALORA`.

---

## Verifying correct experts/heads during eval
CoLA/Hydra already compute routing metrics when `language_prior_weight > 0`:
- Metrics stored in `peft.metrics` via `record_cola_metrics` / `record_hydralora_metrics`.
- Aggregated by `pop_tracked_metrics()`.

In training, these are logged by `SFTTrainer.log`.
In lm_eval, **nothing logs them** by default.

To verify routing during lm_eval:
- Use `scripts/lm_eval_language_ids.py --log-router-metrics`.
- The wrapper calls `pop_tracked_metrics()` after each eval run and writes the metrics into:
  - `results["router_metrics"]` in the output JSON.
  - a pseudo task `_router_metrics` so W&B logs them.

This yields metrics like:
- `cola/language_target_hit_rate`, `hydralora/head_target_*`, `moelpr/expert_*`, etc when language IDs are provided.

---

## Checklist for implementation
- [ ] Confirm adapter has `language_list` in `adapter_config.json`.
- [ ] Implement task → language_id mapping from task name suffix.
- [ ] Run **no‑ID** eval with existing script.
- [ ] Run **with‑ID** eval using wrapper (per‑task or custom LM class).
- [ ] (Optional) Log router metrics with `--log-router-metrics`.

---

## Files referenced
- `configs/lm_eval_tasks.txt`
- `scripts/lm_eval_checkpoint.sh`
- `scripts/tests/lm_eval_checkpoint_local.sh`
- `LLaMA-Factory/src/peft/peft_model.py` (special_peft_forward_args)
- `LLaMA-Factory/src/peft/tuners/cola/model.py` (language_ids forward hooks)
- `LLaMA-Factory/src/peft/tuners/hydralora/model.py` (same pattern)
- `LLaMA-Factory/src/peft/tuners/utils/language_routing.py`
- `LLaMA-Factory/src/peft/metrics.py`
