# Latest checkpoint eval

Minimal helpers to evaluate **only the latest adapter checkpoint** per run directory.
These scripts are designed for large comparison sweeps where each run directory
contains multiple `checkpoint-*` subfolders.

## What this does

- Accepts multiple run paths (each containing `checkpoint-*` folders).
- Resolves the **latest adapter checkpoint** (prefers highest step number).
- Submits a single eval job per run using the local lm-evaluation-harness clone
  and `configs/lm_eval_tasks.txt`.
- Logs results to W&B with stable, structured naming derived from the run path.

## Files

- `submit_latest_eval.py`: discovers latest checkpoints + submits sbatch jobs.
- `lm_eval_harness_latest.sh`: sbatch job script that runs the eval.

## Example

```bash
python scripts/eval/latest_checkpoint/submit_latest_eval.py \
  --paths \
    /scratch/.../llama-3.1-8B_tokenizer-10_langs/adamole/paper-best_20260120_222802 \
    /scratch/.../llama-3.1-8B_tokenizer-10_langs/mola/paper-best_20260120_222802 \
  --wandb-project moe-study-comparison_eval \
  --eval-partition gpu \
  --eval-time 12:00:00 \
  --eval-gpus 1 \
  --eval-gpu-type h100
```

If you already have a list of run paths:

```bash
python scripts/eval/latest_checkpoint/submit_latest_eval.py --paths-file /path/to/run_paths.txt
```

Inline list wrapper (edit the DIRS array inside the script):

```bash
scripts/eval/latest_checkpoint/run_latest_eval_from_list.sh
```

This wrapper auto-selects the tasks file based on the run path:
- `.../10_langs/...` → `configs/lm_eval_tasks_10_langs_with_flores.txt` (if present) or `configs/lm_eval_tasks_10_langs.txt`
- `.../96_langs/...` → `configs/lm_eval_tasks_96_langs_with_flores.txt` (if present) or `configs/lm_eval_tasks_96_langs.txt`
- `.../200_langs/...` → `configs/lm_eval_tasks_200_langs_with_flores.txt` (if present) or `configs/lm_eval_tasks_200_langs.txt`

Important prerequisite:

- FLORES tasks will only run if you’ve generated configs/data in the harness:

```bash
cd scripts/eval/lm-evaluation-harness
python -m lm_eval.tasks.flores_en_perplexity.gen_flores_config
```

## Notes

- The eval job reads tasks from `configs/lm_eval_tasks.txt` by default.
- The job uses the local harness under `scripts/eval/lm-evaluation-harness`.
- Adapter checkpoints are resolved via `adapter_config.json` in the checkpoint
  or `checkpoint-XXXX_adapter`.
- FLORES tasks require generating configs/data in the harness:
   `python -m lm_eval.tasks.flores_en_perplexity.gen_flores_config`
 - `configs/lm_eval_tasks_200_langs_with_flores.txt` is generated from
   `tools/two_stage_clustering/200_tier_language_groupings.json` plus the
   existing FLORES task list.
