# Training Orchestration (Slurm + Checkpoints)

This section describes how training jobs are launched and how checkpoints are monitored and evaluated.

---

For every approach we have a single job script which launches the approach with all necessary params.
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

We then combine these multiple scripts and start them for our ablation study using this main entry script.
- **Launcher**: `scripts/comparison/run_multilingual_ablation.sh`
- **Spec parser**: `scripts/comparison/ablation_specs.py`

Key features:
- Tier-aware resource allocation (GPU count/type, walltime, partition).
- Variant loops for LoRA, Hydra, CoLA.
- Optional automated eval listeners.

---

## 3) Router Validation
This is a sanity check to ensure that we have the correct number of experts and the correct number of A/B matrices before training.
- **Config sanity check**: `scripts/comparison/router_setup.py`
- Validates:
  - `LANGUAGE_MAP` presence
  - expert count vs tier groupings
  - head counts vs subgroup sizes
  - routing mode/guidance scope

---

## 4) Checkpoint Listener + Eval
We suggest not running lm‑eval during the main training run. We only run eval loss during training because it is already wired into the HF trainer. If you run lm‑eval inside training (e.g., via a trainer callback), it gets messy: you need to keep the model on GPU and also launch lm‑eval, which requires extra RAM and causes overhead. There are other disadvantages too.

Instead, save checkpoints during training. The listener script watches for new checkpoints and submits a **separate** lm‑eval job for each checkpoint on another GPU (separate from the training process). This is cleaner: eval can run at full batch size while training fully occupies the training GPUs, and results are cleanly tracked in W&B.
- **Listener**: `scripts/checkpoint_listener.sh`
- **Eval runner**: `scripts/lm_eval_checkpoint.sh`
- **Tasks list**: `configs/lm_eval_tasks.txt`

Flow:
1. Training script emits checkpoints into `OUTPUT_DIR`.
2. Listener watches for `checkpoint-*` and triggers `lm_eval_checkpoint.sh`.
3. Eval logs to W&B under eval project/prefix.


