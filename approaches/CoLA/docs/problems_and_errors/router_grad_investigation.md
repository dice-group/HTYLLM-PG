# Router Gradient Investigation (CoLA / HydraLoRA)

## Summary
During CoLA LPR training, router gradients were initially always reported as zero despite
language prior loss being logged. After allowing router params to remain trainable and adding
diagnostic logging, router gradients became non-zero but sometimes appear partially missing.

## Symptoms
- `language_prior_loss` is logged and `targets_valid` is > 0.
- Router logits show `requires_grad=True` and have a valid `grad_fn`.
- Router cache shows `logits_requires_grad=True` and `targets_valid=True`.
- Earlier: `router_grad_norm=0.000000 routers=0` consistently.
- Now: `router_grad_norm` becomes non-zero, but sometimes `router_grad_none>0`.

## Root Cause Identified
PEFT adapter trainable filtering was freezing router parameters:
- `LLaMA-Factory/src/peft/tuners/cola/model.py::_mark_only_adapters_as_trainable`
- `LLaMA-Factory/src/peft/tuners/hydralora/model.py::_mark_only_adapters_as_trainable`

Both were updated to keep `router` (and `lora_route` in HydraLoRA) trainable.
Specifically, the CoLA filter originally required `self.prefix` and therefore **excluded**
router weights (which are named `*.router.*` and do not include the LoRA prefix). This
meant router params were frozen even when LPR was enabled.

## Current Diagnostics
Logging added to identify autograd connectivity and param grads:
- `LLaMA-Factory/src/peft/tuners/cola/forward.py` prints router logits `requires_grad` and config.
- `LLaMA-Factory/src/llamafactory/train/sft/trainer.py` logs:
  - router cache states
  - logits `requires_grad`
  - `router_grad_norm`, `routers`, `router_grad_none`

## Interpretation of New Logs
When you see:
- `router_grad_norm>0` and `router_grad_none=0` → router params are receiving gradients.
- `router_grad_none>0` → some router params have no gradient for that step. Likely causes:
  - FSDP sharding / gradient materialization per-rank
  - Accumulation step boundary (grads not yet populated)
  - Router parameters unused in the current microbatch (e.g., all tokens routed to a subset of layers)

## Next Checks
1) Confirm the number of router params seen per rank by logging names once.
2) Compare `router_grad_none` across steps and after accumulation boundary.
3) If `router_grad_none` remains high, verify router weights are in optimizer param groups.

## Notes
This doc reflects current state as of 2026-01-08 after enabling router params in trainable
filtering and adding diagnostics. The key fix was ensuring `_mark_only_adapters_as_trainable`
keeps `router` (and Hydra `lora_route`) trainable so LPR gradients can flow.
