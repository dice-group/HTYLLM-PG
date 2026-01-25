# Evaluation and Result Analysis

This section documents how evaluation is run and how results are summarized.

---

## 1) Evaluation Tasks
Eval loss is performed during training.
LM‑eval benchmarks are applied separately on other GPUs through the listener script.
For lm‑eval we focus on **Belebele** and **FLORES** because these benchmarks cover the most languages (122 and 200), while most other multilingual benchmarks only support ~30–40.
Reference: https://arxiv.org/abs/2409.17892
- **Task list**: `configs/lm_eval_tasks.txt`
- **Runner**: `scripts/lm_eval_checkpoint.sh`
- **Listener**: `scripts/checkpoint_listener.sh`


---

## 2) Metrics to Track
During main training we collect extensive metrics. You can check all metrics and their meaning in the file below. Many were added to ensure training stays healthy and to support later analysis.
- **Reference**: `docs/training_metrics.md`
- Key routing metrics:
  - load balance (CV, max/min frac)
  - routing entropy
  - language target hit-rate + neg log-prob
  - auxiliary language prior loss

---

## 3) Result Analysis Artifacts
We created a module where we collect scripts used for analysis.
- **Scripts**: `result_analysis/`
  - Example: `result_analysis/compare_lpr_results_cola_hydralora.py`
- **Outputs**:
  - summary CSVs
  - plots under `result_analysis/paper_eval_summary/`

---

## 4) Expected Inputs
- Training logs + W&B runs
- Checkpoint eval logs from `lm_eval_checkpoint.sh`


## 6) Extended analysis
The module in `approaches/CoLA/Extended_Analysis` creates heatmaps and analyzes/visualizes routing of each language to each layer + expert for analysis.
