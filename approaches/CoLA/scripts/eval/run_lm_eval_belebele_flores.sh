#!/usr/bin/env bash
# Evaluates the hard-coded Llama-3.1-8B model on every Belebele subset, logs to W&B, and saves JSON/CSV summaries.
set -euo pipefail

CKPT="meta-llama/Llama-3.1-8B"
TOKENIZER="meta-llama/Llama-3.1-8B"
OUTDIR="data_prep/processed_artifacts/lm_eval/llama31_8b"

# Discover Belebele languages, otherweise it uses weird varaints such as belebele_amh_prompt_2 or belebele_afr_prompt_5
BELEBELE_TASKS=$(ls lm_eval/tasks/belebele \
  | grep '^belebele_[a-z][a-z][a-z]_' \
  | grep -v 'prompt' \
  | sed 's/.yaml//' \
  | sort -u \
  | paste -sd ',' -)

TASKS="${BELEBELE_TASKS}"

WANDB_PROJECT=${WANDB_PROJECT:-"llama3.1_8b_multilingual_eval"}
WANDB_NAME=${WANDB_NAME:-"lm_eval_belebele_llama31_8b"}
BS=${BS:-"auto"}

mkdir -p "$OUTDIR"
JSON_PATH="${OUTDIR}/lm_eval.json"
CSV_PATH="${OUTDIR}/lm_eval.csv"

echo "Running lm-eval on Belebele tasks: ${TASKS}"

lm_eval \
  --model hf \
  --model_args "pretrained=${CKPT},tokenizer=${TOKENIZER}" \
  --tasks "${TASKS}" \
  --batch_size "${BS}" \
  --output_path "${JSON_PATH}" \
  --wandb_args "project=${WANDB_PROJECT},name=${WANDB_NAME}"


python - <<'PY'
import json
import csv
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

with json_path.open() as f:
    payload = json.load(f)

results = payload.get("results", {})
rows = []
for task, metrics in results.items():
    if not isinstance(metrics, dict):
        continue
    flat_items = metrics.items()
    for subset, values in flat_items:
        if isinstance(values, dict):
            for metric, value in values.items():
                rows.append(
                    {
                        "task": task,
                        "subset": subset,
                        "metric": metric,
                        "value": value,
                    }
                )
        else:
            rows.append(
                {
                    "task": task,
                    "subset": "overall",
                    "metric": subset,
                    "value": values,
                }
            )

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task", "subset", "metric", "value"])
    writer.writeheader()
    writer.writerows(rows)
PY "${JSON_PATH}" "${CSV_PATH}"

echo "lm-eval JSON saved to ${JSON_PATH}"
echo "Flattened CSV saved to ${CSV_PATH}"
