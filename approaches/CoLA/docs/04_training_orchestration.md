# Training Orchestration (Slurm + Checkpoints)

This section describes how training jobs are launched and how checkpoints are monitored and evaluated.

---

## 1) Single-Run Job Scripts
- **CoLA**: `scripts/comparison/cola_lpr_job.sh`
- **HydraLoRA**: `scripts/comparison/hydralora_lpr_job.sh`
- **LoRA baseline**: `scripts/comparison/lora_job.sh`

These scripts:
- Set up conda/module envs and cache dirs.
- Export W&B metadata and routing flags.
- Launch `llamafactory-cli train` or `accelerate launch`.

---

## 2) Multi-Run Ablations
- **Launcher**: `scripts/comparison/run_multilingual_ablation.sh`
- **Spec parser**: `scripts/comparison/ablation_specs.py`

Key features:
- Tier-aware resource allocation (GPU count/type, walltime, partition).
- Variant loops for LoRA, Hydra, CoLA.
- Optional automated eval listeners.

---

## 3) Router Validation
- **Config sanity check**: `scripts/comparison/router_setup.py`
- Validates:
  - `LANGUAGE_MAP` presence
  - expert count vs tier groupings
  - head counts vs subgroup sizes
  - routing mode/guidance scope

---

## 4) Checkpoint Listener + Eval
- **Listener**: `scripts/checkpoint_listener.sh`
- **Eval runner**: `scripts/lm_eval_checkpoint.sh`
- **Tasks list**: `configs/lm_eval_tasks.txt`

Flow:
1. Training script emits checkpoints into `OUTPUT_DIR`.
2. Listener watches for `checkpoint-*` and triggers `lm_eval_checkpoint.sh`.
3. Eval logs to W&B under eval project/prefix.

---

## 5) Canonical Run Plan
- **Docs**: `docs/training_plan.md`
- **CSV schedule**: `docs/training_runs_plan.csv`

Keep these in sync with `run_multilingual_ablation.sh` and the single-run scripts.

