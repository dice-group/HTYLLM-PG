# Evaluation Analysis Guide (Concise)

This guide shows how to aggregate lm-eval results into paper-ready tables and plots.
It is tailored to the 200-language ablation but works for any group with the same
logging structure.

## 1) Run the aggregation script

```
python3 result_analysis/eval_analysis.py \
  --train-project dice-nlp/htyllm-adapter-lpr-200_lang_cola \
  --eval-project dice-nlp/htyllm-adapter-lpr-200_lang_cola_eval \
  --mode with_ids \
  --task-prefix belebele_ \
  --output-dir result_analysis/paper_eval
```

Outputs (CSV + plots) land in `result_analysis/paper_eval`.

## 2) What gets produced

- `result_analysis/paper_eval/per_run_summary.csv`
  - Macro accuracy (mean/median/std).
  - Per-resource means (high/med/low/etc).
  - Router metrics (expert load CV, active frac, entropy, target hit-rate, etc).
  - Training/eval loss for context.
- `result_analysis/paper_eval/per_task_scores.csv`
  - Per-language accuracy with resource and script labels.
- `result_analysis/paper_eval/correlations.csv`
  - Pearson correlations between router metrics and macro accuracy.
- Plots (PNG by default):
  - `overall_accuracy.png` (macro acc by variant)
  - `resource_accuracy.png` (acc by resource tier)
  - `router_vs_router_*` (router metric vs acc)

## 3) Key analysis questions (ACL-ready)

1) **Resource-tier gains**: aggregate Belebele acc by resource bucket
   (high/med/low). Check if routing helps low-resource more than high-resource.
2) **Routing alignment vs quality**: correlate `language_target_hit_rate`,
   `language_target_neglogp`, and `router_entropy` with macro acc to quantify
   whether better alignment leads to better accuracy.
3) **Router health**: use `expert_load_cv`, `active_expert_frac` to detect
   collapse and relate to performance drops.
4) **Stability vs accuracy**: compare std of per-language accuracy for LPR
   vs non-LPR variants (does routing reduce variance across languages?).
5) **Loss context**: report `train/loss` and `eval/loss` alongside accuracy
   to show whether routing affects optimization quality or only evaluation.

## 4) Notes / troubleshooting

- `--mode with_ids` is required for per-language scores. If a run has no
  with-ids evals, it will report `task_count=0`.
- You can switch to `--mode no_ids` or `--mode any`, but those do not provide
  per-language breakdowns unless tasks are logged in summary keys.
- The resource buckets are read from
  `data_prep/base_data/lang_resource_dataset.tsv`.

## 5) Plot customization

Add formats:

```
--plot-formats png,pdf
```

Adjust task selection:

```
--task-prefix belebele_
```

For other eval suites, change the prefix or re-map tasks and update the
resource map accordingly.
