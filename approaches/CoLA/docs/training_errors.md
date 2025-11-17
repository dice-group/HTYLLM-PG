# Training Errors

## Error 1 – DDP parameter marked ready twice
- Command/script: `accelerate_moe_cola_train.sh` using `LLaMA-Factory/examples/accelerate/ddp_4gpu_config.yaml`.
- Hardware/setup: 4×H100 via Slurm, bf16, CoLA LoRA experts (`cola_num_experts=4`, `cola_top_k=1`, checkpointing enabled by default).
- Failure: `RuntimeError: Expected to mark a variable ready only once` for `base_model.model.model.layers.31.mlp.down_proj.lora_B.expert_3.3.weight` during the backward pass, raised by Accelerate/DDP.
- Trace shows the error originates inside `torch.utils.checkpoint` re-entering backward, implying a shared parameter is used across multiple reentrant backward passes.
- Run metadata: WANDB run `cola_moe_acc_5langs_20251115_133015`; training aborts before completing the first evaluation step.
- Root cause: CoLA enables multiple experts inside each LoRA layer, and Hugging Face enables gradient checkpointing with `use_reentrant=True` by default. Each transformer block is wrapped by `torch.utils.checkpoint`, so the same expert weights are re-used in multiple reentrant backward passes, which vanilla DDP forbids.
- Fix (confirmed): keep gradient checkpointing for memory savings but switch Accelerate to the non-reentrant checkpoint path by passing `--use_reentrant_gc False`. `accelerate_moe_cola_train.sh` now defines `USE_REENTRANT_GC=False` and forwards it to `train.py`, which prevents DDP from seeing duplicate “ready” hooks.

## Error 2 – FSDP auto-wrap mixed dtype flatten failure
- Command/script: `accelerate_moe_cola_train.sh` but with `ACCEL_CONFIG=./LLaMA-Factory/examples/accelerate/fsdp_4gpu_config.yaml`.
- Hardware/setup: identical 4×H100 Slurm node, bf16 requested, CoLA experts enabled.
- Failure: Accelerate hands the model to PyTorch FSDP; during auto wrapping, FSDP raises `ValueError: Must flatten tensors with uniform dtype but got torch.bfloat16 and torch.float32` while building a `FlatParamHandle`.
- Stack trace shows the error occurs before training starts, when FSDP tries to flatten all parameters inside each wrapped module.
- Likely cause: CoLA inserts additional adapter parameters (router, lora_B, etc.) whose dtype stays `float32` while the base model modules are already cast to bf16; FSDP requires that every wrapped module contains parameters with a single dtype, so it refuses to flatten mixed bf16/fp32 tensors.
- Status: Even after forcing `pure_bf16=true` and casting adapter weights to bf16 inside `_setup_cola_tuning`, FSDP still reports mixed bf16/fp32 tensors when flattening. Root cause remains unresolved and needs deeper inspection of when/how certain adapter buffers are allocated. (Re-run logs from 2025‑11‑15 15:23 show the failure persists.)
