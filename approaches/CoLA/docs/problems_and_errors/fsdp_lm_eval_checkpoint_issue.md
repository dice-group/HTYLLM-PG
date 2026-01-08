# FSDP Checkpoints Not Loadable for LM-Eval (CoLA/Hydra)

## Problem Summary
With `LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml`, checkpoints are saved in an FSDP layout that
`scripts/lm_eval_checkpoint.sh` cannot load directly. The eval script only knows how to:
- Load a standard HF `save_pretrained` directory (`config.json` + model weights), or
- Load a PEFT adapter directory (`adapter_config.json` + adapter weights).

The raw FSDP checkpoints do not include `config.json` or `adapter_config.json`, so eval fails unless a matching
`checkpoint-XX_adapter/` exists (or you export a full HF checkpoint).

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

## CoLA/Hydra Router Weights (Status)
Older builds filtered adapter weights strictly by keys containing `"lora_"`, which dropped CoLA/Hydra expert routers
(`*.router.*`). This repo now explicitly includes `.router.` keys for `COLA` and `HYDRALORA`, so adapter checkpoints
should retain expert router weights. If you still see missing router weights, backport the router inclusion in
`save_and_load.py`.

File references:
- `LLaMA-Factory/src/peft/utils/save_and_load.py` (`get_peft_model_state_dict` includes `.router.` for COLA/HYDRALORA)
- `LLaMA-Factory/src/peft/tuners/cola/layer.py` (expert router lives in `self.router`)
- `LLaMA-Factory/src/peft/tuners/hydralora/layer.py` (expert router lives in `self.router`)

## Existing In-Repo Solutions
1) Save adapter checkpoints during training
   - Callback: `SaveAdapterCheckpointCallback` in
     `LLaMA-Factory/src/llamafactory/train/callbacks.py`
   - Creates `checkpoint-XXXX_adapter/` containing `adapter_config.json`
     and adapter weights.

2) Eval scripts already prefer adapters
   - `scripts/lm_eval_checkpoint.sh` and `scripts/checkpoint_listener.sh` will
     auto-swap to `checkpoint-*_adapter/` if present.

3) Export a full HF checkpoint (optional)
   - `llamafactory-cli export` can merge base + adapter into a standalone
     `save_pretrained` directory for evals that need a full checkpoint.

## How We Should Continue (Recommended Path)
1) Confirm whether adapter checkpoints exist
   - Check for `checkpoint-XX_adapter/` alongside FSDP checkpoint dirs.
2) If adapter checkpoints are missing
   - Re-run training (preferred) or manually export via `llamafactory-cli export`
     once an adapter is available.
3) If the custom callback is failing
   - Inspect training logs for errors from `SaveAdapterCheckpointCallback`.
   - Ensure the training model is a `PeftModel` wrapped in FSDP.
4) Fix adapter saving for CoLA/Hydra routers
   - Verify `get_peft_model_state_dict()` includes `.router.` keys for
     `COLA`/`HYDRALORA` (older branches may still drop them).
5) Optional: update checkpoint listener
   - Prefer evaluating `checkpoint-*_adapter/` if present.
   - Otherwise, export a full HF checkpoint before eval.

## Practical Options for This Repo
1) Prefer adapter checkpoints for eval
   - Ensure `checkpoint-*_adapter/` is written (with router weights saved).
2) Export a full HF checkpoint when needed
   - Use `llamafactory-cli export` to merge base + adapter into a
     `save_pretrained` directory for `lm_eval`.
3) Save full HF checkpoint at train time (if you need it)
   - Call `save_pretrained` with a full state dict (or use `llamafactory-cli export`)
     so checkpoints are loadable without conversion.
