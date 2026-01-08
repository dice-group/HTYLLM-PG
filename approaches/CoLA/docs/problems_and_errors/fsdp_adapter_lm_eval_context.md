# FSDP Adapter Checkpoints + LM-Eval Context (CoLA / HydraLoRA)

## Problem Summary
Distributed (FSDP) training saves **full model checkpoints** that are not directly usable for LM-eval when we only want **adapter weights** (CoLA/HydraLoRA). A custom callback creates `checkpoint-*_adapter` directories, but:

- FSDP `state_dict()` is a **collective** (all ranks must participate). If non‑rank0 returns early, it can hang or time out.
- Full-state all-gather is **expensive** and stalls training when adapters are saved too often.
- Without adapter checkpoints, lm-eval needs full `model.safetensors` + FSDP shards or fails.

We now save **adapter checkpoints** alongside FSDP checkpoints to enable **fast eval** with the base model.

## Current Checkpoint Layout
**Distributed / FSDP checkpoint (example):**
```
checkpoint-40/
  model.safetensors
  pytorch_model_fsdp.bin
  optimizer.bin
  rng_state_*.pth
  scheduler.pt
  trainer_state.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  training_args.bin
```
**Adapter checkpoint:**
```
checkpoint-40_adapter/
  adapter_model.safetensors
  adapter_config.json
  README.md
```

The adapter folder is the one used for LM-eval with PEFT.

## How Adapter Saving Works (Current Fix)
Callback: `LLaMA-Factory/src/llamafactory/train/callbacks.py`

- Uses PyTorch **distributed checkpoint API**:
  - `torch.distributed.checkpoint.state_dict.get_model_state_dict`
  - `StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)`
- All ranks participate in the collective; **rank0 only** writes files.
- Strips `_fsdp_wrapped_module.` prefix before filtering adapter keys.
- Ensures `.router.` and `.expert_` parameters are captured (CoLA/Hydra custom modules).

**Why this matters:**
- `ignore_frozen_params=True` avoids gathering frozen base weights.
- Only trainable params are gathered, reducing size and time.

## Resolution: Adapter-Only FSDP Save (Current Approach)
We now prefer the **official adapter-only FSDP save path** (Transformers PR #28297) instead of manually extracting
adapter keys from sharded state dicts.

Implementation summary:
- In `SaveAdapterCheckpointCallback` (FSDP case), we call:
  - `accelerate.utils.fsdp_utils.save_fsdp_model(..., adapter_only=True)`
- We **temporarily switch** the FSDP plugin to `FULL_STATE_DICT` with matching configs:
  - `state_dict_type = FULL_STATE_DICT`
  - `state_dict_config = FullStateDictConfig(offload_to_cpu, rank0_only)`
  - `optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu, rank0_only)`
  - Then restore the original configs after saving.
- The adapter-only save produces a single state dict file in `checkpoint-XX_adapter/`.
  We immediately re-save it as a standard PEFT adapter (`adapter_model.safetensors`) and remove the temporary file.

Why this fixes the NCCL timeouts:
- Avoids manual sharded-key filtering (which can diverge across ranks).
- Uses the upstream, synchronized adapter-only save path (single collective).
- Prevents mismatched collectives that previously caused `_ALLGATHER_BASE` timeouts.

Relevant code:
- `LLaMA-Factory/src/llamafactory/train/callbacks.py` (adapter-only FSDP path + config swap)
- Workflow injection of accelerator into the callback (so it can call `save_fsdp_model`)


## LM-Eval Integration (with/without language IDs)
We use a custom wrapper script to run two evals: **with language IDs** and **without**.

**Wrapper:** `scripts/lm_eval_language_ids.py`
- `--mode with_ids|no_ids|both`
- Optional router metrics: `--log-router-metrics`
- Accepts `--torch-dtype` and `--device-map` (or env `LM_EVAL_TORCH_DTYPE`, `LM_EVAL_DEVICE_MAP`)
- Writes two result JSONs:
  - `with_language_ids_*.json`
  - `no_language_ids.json`

**Cluster auto-eval:** `scripts/lm_eval_checkpoint.sh`
- Watches for new checkpoints and runs lm-eval
- Uses adapter checkpoint if `language_list` exists in adapter config

**Local eval:** `scripts/tests/lm_eval_checkpoint_local.sh`

## Log Locations
- Listener logs: `scripts/comparison/logs/multilingual_ablation/<tier>/eval/`
- LM-eval runs output to `.../eval/logs/` (timestamped)

## Common Errors & Fixes

### 1) FSDP Timeout / All-gather Stall
**Cause:** Full-state gather is expensive. If any rank doesn’t reach the collective, it hangs.
**Mitigations:**
- Reduce `save_steps` / save frequency
- Ensure all ranks enter `get_model_state_dict` (no early returns)
- Save to local SSD if available
- Consider sharded adapter save (advanced; not implemented)

### 2) Eval missing dataset
```
ValueError: eval_strategy=steps but no eval_dataset
```
**Fix:** set `EVAL_STRATEGY=no` in local scripts or provide eval dataset.

### 3) FlashAttention GLIBC error
```
ImportError: GLIBC_2.32 not found
```
**Fix:** disable FA2: `FLASH_ATTN=disabled`

### 4) Tokenization error: `language` field
```
ValueError: too many dimensions 'str' / language is list
```
**Fix:** tokenizer must drop text + raw language fields and only keep `language_ids`/`family_ids`.

## Local Smoke Test Setup

### Tokenize preview subset
`scripts/local/tokenize_preview_subset.py`
- Reads: `/data/project_data/moe_study/fw_samples/preview_subset/*.jsonl.gz`
- Writes: `/data/project_data/moe_study/tokenized/preview_subset_tiny_llama`
- Model: `hf-internal-testing/tiny-random-LlamaForCausalLM`

### Run 2‑GPU local FSDP training
`scripts/local/run_cola_lpr_local_2gpu.sh`
- Uses `MAX_STEPS=100`, `SAVE_STEPS=40`, `EVAL_STRATEGY=no`
- Saves adapter checkpoints and runs local lm-eval

### Example LM-eval (manual)
```
LM_EVAL_TORCH_DTYPE=bf16 LM_EVAL_DEVICE_MAP=auto \
python3 scripts/lm_eval_language_ids.py \
  --checkpoint /path/to/checkpoint-4 \
  --tokenizer meta-llama/Llama-3.2-1B \
  --tasks belebele_zul_Latn \
  --output-dir /tmp/lm_eval_smoke \
  --batch-size 1 --limit 5 --mode both \
  --log-router-metrics \
  --wandb-args "project=htyllm-lm-eval-smoke,name=cola_lpr_checkpoint4"
```

## Known Limitations
- Adapter saving still requires **collective sync** across ranks → can pause training.
- `ignore_frozen_params=True` saves trainables only; any trainable weights not matched by PEFT rules must be explicitly included.
- Router metrics are only meaningful when `language_ids` are passed.

## Possible Upstream Option (Not Implemented)
Hugging Face Transformers PR **#28297** adds `adapter_only` support to the FSDP save/load path so Trainer can persist
only PEFT adapters instead of full model shards. This could avoid heavy all‑gathers when saving under FSDP, but our
current adapter checkpoints are created by a **custom callback** (not by the Trainer FSDP save path), so this PR does not
directly fix our current HydraLoRA save timeout. It is a **possible future option** if we decide to refactor adapter
checkpointing to use the upstream FSDP adapter‑only save API.

## Files to Know
- Adapter save callback: `LLaMA-Factory/src/llamafactory/train/callbacks.py`
- PEFT save/load logic: `LLaMA-Factory/src/peft/utils/save_and_load.py`
- CoLA layer: `LLaMA-Factory/src/peft/tuners/cola/layer.py`
- HydraLoRA layer: `LLaMA-Factory/src/peft/tuners/hydralora/layer.py`
- Auto-eval: `scripts/lm_eval_checkpoint.sh`
- Wrapper: `scripts/lm_eval_language_ids.py`
- Local guide: `scripts/local/LOCAL_RUN_GUIDE.md`
