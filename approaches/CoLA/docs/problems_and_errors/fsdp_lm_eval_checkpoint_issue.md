# FSDP Checkpoints Not Loadable for LM-Eval (CoLA/Hydra)

## Problem Summary
With `LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml`, checkpoints are saved in an FSDP layout that
`scripts/lm_eval_checkpoint.sh` cannot load directly. The eval script only knows how to:
- Load a standard HF `save_pretrained` directory (`config.json` + model weights), or
- Load a PEFT adapter directory (`adapter_config.json` + adapter weights).

The FSDP checkpoints do not include `config.json`, and adapter checkpoints may be missing or incomplete, so eval fails.

## Observed Layout (Example)
```
checkpoint-40/
  chat_template.jinja
  model.safetensors
  optimizer.bin
  pytorch_model_fsdp.bin
  rng_state_0.pth
  rng_state_1.pth
  rng_state_2.pth
  rng_state_3.pth
  scheduler.pt
  special_tokens_map.json
  tokenizer_config.json
  tokenizer.json
  trainer_state.json
  training_args.bin
```
Notably missing: `config.json`, `adapter_config.json`.

## CoLA/Hydra-Specific Root Cause
In PEFT, `get_peft_model_state_dict()` filters LoRA-family adapters by keys containing "lora_". This means:
- Expert routers in CoLA/Hydra are named `router.*` (no `lora_` prefix) and are not saved.
- Head routers in Hydra use `lora_route.*` and are saved.

Result: even if an adapter checkpoint exists, expert router weights are lost, which breaks routing at eval time.

File references:
- `LLaMA-Factory/src/peft/utils/save_and_load.py` (`get_peft_model_state_dict`)
- `LLaMA-Factory/src/peft/tuners/cola/layer.py` (expert router lives in `self.router`)
- `LLaMA-Factory/src/peft/tuners/hydralora/layer.py` (expert router lives in `self.router`)

## Existing In-Repo Solutions
1) Save adapter checkpoints during training
   - Callback: `SaveAdapterCheckpointCallback` in
     `LLaMA-Factory/src/llamafactory/train/callbacks.py`
   - Should create `checkpoint-XXXX_adapter/` containing `adapter_config.json`
     and adapter weights.

2) Export FSDP checkpoints to adapter format
   - Script: `scripts/comparison/export_fsdp_checkpoint.py`
   - Batch wrapper: `scripts/comparison/export_fsdp_checkpoints.sh`
   - These detect FSDP layouts and export a loadable adapter directory.

## How We Should Continue (Recommended Path)
1) Confirm whether adapter checkpoints exist
   - Check for `checkpoint-XX_adapter/` alongside FSDP checkpoint dirs.
2) If adapter checkpoints are missing, use the export script
   - Export `checkpoint-XX` to `checkpoint-XX_adapter` and point lm-eval at it.
3) If the custom callback is failing
   - Inspect training logs for errors from `SaveAdapterCheckpointCallback`.
   - Ensure the training model is a `PeftModel` wrapped in FSDP.
4) Fix adapter saving for CoLA/Hydra routers
   - Extend `get_peft_model_state_dict()` to include `router.*` keys when
     `peft_type` is `COLA` or `HYDRALORA`, so expert routers are saved.
5) Optional: update checkpoint listener
   - Prefer evaluating `checkpoint-*_adapter/` if present.
   - Otherwise, call `export_fsdp_checkpoint.py` before eval.

## Practical Options for This Repo
1) Prefer adapter checkpoints for eval
   - Ensure `checkpoint-*_adapter/` is written (with router weights saved).
2) Convert FSDP checkpoints before eval
   - Use `scripts/comparison/export_fsdp_checkpoint.py` (or Accelerate
     `merge_fsdp_weights`) to produce a loadable directory for `lm_eval`.
3) Save full HF checkpoint at train time
   - Force `FULL_STATE_DICT` and call `save_pretrained` with
     `accelerator.get_state_dict(model)` so checkpoints are loadable without conversion.
