# W&B Summary Aggregation Problem (lm-eval)

## Context and ground truth

We run lm-eval on checkpoints produced by FSDP training (CoLA / HydraLoRA). The training jobs
save two kinds of artifacts for each checkpoint step:

1) FSDP checkpoint shards (used to resume training)
- Example: `checkpoint-40/` contains `pytorch_model_fsdp_0`, `optimizer_0`, `rng_state_*`, etc.
- This is not directly loadable by lm-eval.

2) Adapter artifacts (used for evaluation)
- Sharded adapter (fast save during FSDP): `checkpoint-40_adapter_sharded/`
  - Files: `__0_0.distcp`, `__1_0.distcp`, `adapter_config.json`, `adapter_sharded.json`.
- Merged adapter (loadable by PEFT): `checkpoint-40_adapter/`
  - Files: `adapter_config.json`, `adapter_merge.json`, `adapter_model.safetensors`, `README.md`.

We want lm-eval to load the base model + adapter checkpoint for evaluation.

## Evaluation pipeline (current design)

1) Listener detects a completed checkpoint.
2) `scripts/merge_adapter_shards.py` merges `*_adapter_sharded/` into `*_adapter/`.
3) `scripts/lm_eval_language_ids.py` runs evaluation on `checkpoint-XX_adapter`.
4) Eval outputs JSON per mode:
   - No language ids: `no_language_ids.json`
   - With language ids: `with_language_ids_<task>.json`

These eval outputs are written under an output directory:

- Desired structure:
  - `<OUT>/all/<RUN_NAME>/<CHECKPOINT_DIR>/no_language_ids.json`
  - `<OUT>/all/<RUN_NAME>/<CHECKPOINT_DIR>/with_language_ids_<task>.json`

Where:
- `RUN_NAME` = training run name (e.g., `cola_colaflat_20260106_173545`)
- `CHECKPOINT_DIR` = checkpoint folder name (e.g., `checkpoint-40_adapter`)

## W&B logging expectations

We want two distinct W&B surfaces:

1) **Detailed runs** (per checkpoint, per mode, per task):
   - These are already logged by `scripts/lm_eval_language_ids.py`.
   - Example run names:
     - `cola_colaflat_20260106_173545_ckpt40_no_ids_detailed`
     - `cola_colaflat_20260106_173545_ckpt40_with_ids_belebele_eng_Latn_detailed`

2) **Summary runs** (aggregated across checkpoints, one run per mode):
   - Exactly two W&B runs per training run:
     - `<RUN_NAME>_no_ids`
     - `<RUN_NAME>_with_ids`
   - Each run accumulates **one step per checkpoint** (step = checkpoint number).
   - Every task should appear as a metric in those runs:
     - e.g., `belebele_eng_Latn/acc` over steps 40, 80, 120, ...
   - This should yield line plots with one series per mode.

## The problem

We repeatedly see **duplicate summary runs** and **timeouts** because:

- Summary job reads from an output directory that contains results from multiple checkpoints.
  - This causes re-processing of all prior tasks/results in one job.
  - For large task sets, this is slow and can time out (or block W&B init).

- Summary runs are created **per checkpoint** because the W&B run ID is stored in a
  checkpoint-specific output dir. This produces multiple runs with the same name, but
  different IDs, so no line plot across checkpoints.

- When output dirs are not per-checkpoint, summary job mixes checkpoints and can also
  overwrite earlier results or re-log the same metrics many times.

## Current observed state (from latest run)

- Mixed outputs exist at multiple levels:
  - Run root `.../lm_eval_smoke/all/` contains legacy `no_language_ids.json`,
    `with_language_ids_*.json`, and `.wandb_summary_id_*` files.
  - Run subdir `.../all/<RUN_NAME>/` also contains JSONs and `.wandb_summary_id_*`.
  - Per-checkpoint dirs `.../all/<RUN_NAME>/checkpoint-XX_adapter/` contain the
    correct per-checkpoint JSONs.
- Summary ID files exist in both run root and checkpoint dirs:
  - This produces duplicate W&B runs and prevents a single line plot across checkpoints.
- Concurrent summary jobs (ckpt-40 and ckpt-80) try to resume the same W&B run ID:
  - W&B init hangs at `setting up run summary_*` due to lock contention.

## Ground-truth behavior we want

1) **Per-checkpoint output dir**
   - Each checkpoint’s eval output must live in its own folder.
   - This ensures summary jobs only read that checkpoint’s JSON files.

2) **Stable summary run IDs**
   - For a given training run name, summary logging must re-use the same W&B run IDs
     across checkpoints (one for `no_ids`, one for `with_ids`).
   - This yields a single line per mode across checkpoint steps.

3) **No cross-checkpoint reprocessing**
   - Summary job should read only the JSONs for the current checkpoint.
   - It should not scan all checkpoints under the run root.

4) **Clear output separation**
   - Output tree should make it obvious which checkpoint a JSON belongs to.
   - Example (desired):
     - `/.../lm_eval_smoke/all/cola_colaflat_20260106_173545/checkpoint-40_adapter/with_language_ids_belebele_eng_Latn.json`
     - `/.../lm_eval_smoke/all/cola_colaflat_20260106_173545/checkpoint-80_adapter/with_language_ids_belebele_eng_Latn.json`

5) **W&B summary runs remain constant**
   - Only two summary runs should exist per training run:
     - `cola_colaflat_20260106_173545_no_ids`
     - `cola_colaflat_20260106_173545_with_ids`
   - Each new checkpoint appends metrics at step `checkpoint-N`.

## Where the data lives (example)

- Training run root:
  - `/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260106_173545/`

- Adapter checkpoint:
  - `checkpoint-40_adapter/adapter_model.safetensors`

- Eval output dir (desired per checkpoint):
  - `/scratch/hpc-prf-merlin/project_data/moe_study/lm_eval_smoke/all/cola_colaflat_20260106_173545/checkpoint-40_adapter/`
  - JSONs:
    - `no_language_ids.json`
    - `with_language_ids_belebele_eng_Latn.json`
    - `with_language_ids_belebele_deu_Latn.json`
    - `with_language_ids_belebele_zul_Latn.json`

## What we want to fix (next step)

- Ensure eval outputs are **per-checkpoint**.
- Ensure summary job reads only the current checkpoint’s output directory.
- Ensure summary runs use **stable IDs** shared across checkpoints for the same run name.
- Avoid reprocessing or re-logging old checkpoint JSON files in a single summary step.

## Fix design (target behavior)

1) **Enforce per-checkpoint summary input**
   - Summary job should refuse output dirs that are not a checkpoint dir.

2) **Stable run IDs at the run root**
   - Store `.wandb_summary_id_no_ids` and `.wandb_summary_id_with_ids` under
     `.../all/<RUN_NAME>/`, not under the checkpoint dir.

3) **Serialize summary updates**
   - Add a file lock in the run root (e.g., `.wandb_summary_lock`) so only one summary
     job updates the shared W&B runs at a time.

4) **Keep per-checkpoint output dirs**
   - `.../all/<RUN_NAME>/checkpoint-XX_adapter/` is the only place summary should read JSONs.
