# Compute-Test Profiling Summary

This note documents the FLOP profiling runs launched via `scripts/launch_accelerate_moe_cola_pair.sh` with `RUN_LABEL=compute_test`. Two Accelerate jobs were queued: a PiSSA-initialized CoLA adapter stack and a non-PiSSA baseline. Both wrap the same backbone (`meta-llama/Llama-3.1-8B`) and were profiled before training through `scripts/stat_utils/cal_flops.py`.

## Environment Snapshot
- **Hardware/Runtime:** Each job ran on a 4×H100 node under the `gpu1020` partition with CUDA available (`docs/train_compute_test_pissa_330027.txt:1-8`, `docs/train_compute_test_nopissa_330028.txt:1-8`).
- **Model:** The profiler loaded the standard 8B llama3.1 config (4096-dim hidden state, 32 layers, 32 attention heads) with 8.03 B parameters and FlashAttention in “auto” mode (`docs/train_compute_test_pissa_330027.txt:10-67`, `docs/train_compute_test_pissa_330027.txt:165-167`).
- **Input Shape:** Profiling used the same arguments as training (per-device batch size 16, `cutoff_len=2048`), so the measured FLOPs reflect the actual SFT dataloader footprint.

## FLOP Measurements

| Run | Params / GPU | Forward MACs / GPU | Forward FLOPs / GPU | Forward Latency | Sustained FLOPs / GPU |
| --- | --- | --- | --- | --- | --- |
| PiSSA CoLA | 8.03 B | 127.35 TMACs | 255.86 TFLOPs | 628.25 ms | 407.27 TFLOPs/s (`docs/train_compute_test_pissa_330027.txt:177-190`) |
| Non-PiSSA CoLA | 8.03 B | 127.35 TMACs | 255.86 TFLOPs | 627.17 ms | 407.96 TFLOPs/s (`docs/train_compute_test_nopissa_330028.txt:177-195`) |

Additional breakdowns (per-layer latency, MAC share) appear right after the summary in each log (e.g., `docs/train_compute_test_nopissa_330028.txt:185-220`).

## Interpretation
1. **Identical Backbone Cost:** LoRA/CoLA adapters introduce <1 % additional parameters, so both runs sit at the same 8.03 B parameters and ~2.6×10¹⁴ forward FLOPs per GPU. PiSSA modifies initialization only, so it does not move compute.
2. **Throughput Health:** Each profile reports 255.86 TFLOPs of forward work completed in ~0.63 s (`docs/train_compute_test_pissa_330027.txt:177-184` and `docs/train_compute_test_nopissa_330028.txt:177-184`), so the sustained rate is `255.86 / 0.628 ≈ 4.07 × 10² TFLOPs/s` per GPU—about 41% of an H100 SXM BF16 peak (~989 TFLOPs, derived from NVIDIA’s 1,979 TFLOPs sparse figure at FP16/BF16 [lenovo source](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu)). That placement (mid-30s to low-40s percent) matches expectations once SDPA, gradient checkpointing, and host sync overheads are included; significantly higher efficiency would require larger batches/longer sequences or more aggressive kernel fusion rather than adapter changes.
3. **Scaling Guidance:** FLOPs scale linearly with `per_device_train_batch_size` and `cutoff_len`. If you need a lighter compute footprint (e.g., to avoid OOM), reduce one of those knobs rather than toggling PiSSA. Conversely, higher sequence lengths will quickly push the FLOP budget and memory upward; consult this table when planning longer-context experiments.

## Throughput Calculation Check
1. **Profiler Inputs:** Each log reports forward work of 255.86 TFLOPs completed in roughly 0.628 s per GPU (`docs/train_compute_test_pissa_330027.txt:177-184`, `docs/train_compute_test_nopissa_330028.txt:177-184`).
2. **Sustained Rate:** `255.86 TF / 0.628 s ≈ 4.07 × 10² TFLOPs/s`, i.e., ~407 TFLOPs/s per GPU.
3. **Peak Reference:** Lenovo’s H100 PCIe/SXM brief quotes a 1,979 TFLOPs BF16 peak with sparsity [link](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu); halving for the non-sparse case yields ~989 TFLOPs.
4. **Utilization:** `407 / 989 ≈ 0.41`, so the run sits at ~41% of theoretical non-sparse peak—typical once SDPA kernels, activation checkpointing, and host overheads are considered.

## Early Throughput Observations
Both runs were launched with `--include_num_input_tokens_seen true`, so `trainer_log.jsonl` now carries `throughput` (tokens/sec) alongside the usual metrics. The first few checkpoints already show a clear gap:

| Run | Step 1 throughput | Step 10 throughput | Step 20 throughput | Notes |
| --- | --- | --- | --- | --- |
| PiSSA CoLA (2A / 4B) | 5,017 tok/s | 5,730 tok/s | 5,775 tok/s | `/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_compute_test/pissa/trainer_log.jsonl` |
| Non-PiSSA baseline (1A / 1B) | 7,413 tok/s | 8,170 tok/s | 8,217 tok/s | `/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_compute_test/nopissa/trainer_log.jsonl` |

The baseline job uses only one A/B matrix pair, whereas the “PiSSA” job stacks 2×A and 4×B matrices per CoLA block. That extra routing and adapter work explains the ~40% throughput penalty despite identical backbone FLOPs—PiSSA itself is just an initialization method and does not add runtime cost. For a fair PiSSA vs non-PiSSA comparison, keep `NUM_A`/`NUM_B` constant across runs; use higher matrix counts only when you explicitly need the extra adapter capacity.

## Next Steps
1. Track `throughput` and `effective_tokens_per_sec` from the live training logs (`trainer_log.jsonl` and `train_results.json`) once the runs complete to see whether realized tokens/s match the ~0.63 s forward-pass timing.
2. If you intend to benchmark other adapter layouts (different `num_A/num_B`, expert counts, or MoE top‑k), reuse this profiling workflow to confirm they do not inflate FLOPs unexpectedly.
3. Consider pinning `TRITON_CACHE_DIR` to a local SSD (the profiler emitted an NFS warning at `docs/train_compute_test_pissa_330027.txt:9`) to avoid cache stalls during repeated runs.
