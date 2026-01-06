# Local 2-GPU CoLA LPR: Task + Problem Summary

## Task
Run a local 2-GPU FSDP CoLA LPR training with adapter checkpoints and eval (with/without language IDs), without Slurm.

## Current Problem
Distributed adapter checkpoints previously failed to save correctly under FSDP (missing LoRA keys / eval deadlocks). We now save **sharded adapter checkpoints** during training and merge them offline before eval:
- sharded adapter checkpoints under `checkpoint-*_adapter_sharded/`
- merged adapter checkpoints under `checkpoint-*_adapter/`
- run lm-eval with and without language IDs
- keep the loop KISS and reproducible for local debugging

## Relevant Scripts
- Local runner: `scripts/local/run_cola_lpr_local_2gpu.sh`
- Local tokenizer (optional): `scripts/local/tokenize_preview_subset.py`
- Local eval runner: `scripts/tests/lm_eval_checkpoint_local.sh`
- Local checkpoint listener: `scripts/tests/checkpoint_listener_local.sh`
- CoLA training job (used by local runner): `scripts/comparison/cola_lpr_job.sh`

## Key Fixes / References
- Adapter save for FSDP (sharded): `LLaMA-Factory/src/llamafactory/train/callbacks.py` (SaveAdapterCheckpointCallback)
- Merge sharded adapters: `scripts/merge_adapter_shards.py`
- Adapter state dict filtering: `LLaMA-Factory/src/peft/utils/save_and_load.py`
- LM-eval wrapper with language IDs: `scripts/lm_eval_language_ids.py`

## How to Run (Tokenized Data Already Exists)
```
cd /upb/users/j/joeldag/profiles/unix/cs/HTYLLM-PG/approaches/CoLA
bash scripts/local/run_cola_lpr_local_2gpu.sh
```

## Expected Outputs
- Checkpoints: `outputs/local_cola_lpr_2gpu/<run>/checkpoint-*`
- Sharded adapter checkpoints: `outputs/local_cola_lpr_2gpu/<run>/checkpoint-*_adapter_sharded/`
- Merged adapter checkpoints: `outputs/local_cola_lpr_2gpu/<run>/checkpoint-*_adapter/adapter_model.safetensors`
- Eval outputs: `outputs/local_cola_lpr_2gpu/<run>/lm_eval/*.json`

## Quick Verification
```
ls -la outputs/local_cola_lpr_2gpu/*/checkpoint-*_adapter
```

If the adapter is correct, it should contain:
`adapter_config.json`, `adapter_model.safetensors`, `README.md`.
