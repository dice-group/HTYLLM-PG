# Evaluation and Result Analysis

This section documents how evaluation is run and how results are summarized.

---

## 1) Evaluation Tasks
- **Task list**: `configs/lm_eval_tasks.txt`
- **Runner**: `scripts/lm_eval_checkpoint.sh`
- **Listener**: `scripts/checkpoint_listener.sh`

Evaluation is typically triggered automatically by the checkpoint listener during training.

---

## 2) Metrics to Track
- **Reference**: `docs/training_metrics.md`
- Key routing metrics:
  - load balance (CV, max/min frac)
  - routing entropy
  - language target hit-rate + neg log-prob
  - auxiliary language prior loss

---

## 3) Result Analysis Artifacts
- **Scripts**: `result_analysis/`
  - Example: `result_analysis/compare_lpr_results_cola_hydralora.py`
- **Outputs**:
  - summary CSVs
  - plots under `result_analysis/paper_eval_summary/`

---

## 4) Expected Inputs
- Training logs + W&B runs
- Checkpoint eval logs from `lm_eval_checkpoint.sh`

