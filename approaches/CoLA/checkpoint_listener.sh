#!/bin/bash
set -euo pipefail

usage() {
cat <<EOF
Usage: checkpoint_listener.sh --watch-dir DIR --eval-script SCRIPT [options]

Automatically detects new checkpoints in a directory and submits sbatch
evaluation jobs for each one.

Options:
  --watch-dir DIR      Directory containing checkpoint-* folders
  --eval-script SCRIPT Script run via sbatch for evaluation
  --tasks LIST         lm-eval tasks (default: belebele)
  --batch-size N       lm-eval batch size (default: auto)
  --output-dir DIR     Where to save eval outputs (default: watch-dir/lm_eval)
  --wandb-project NAME W&B project (default: llama31_multilingual_eval_belebele)
  --wandb-prefix PREF  W&B prefix (default: cola_moe_acc)
  --extra-args "ARGS"  Extra args for lm_eval
  --poll-interval SEC  Scan interval (default: 120)
  --state-file FILE    Track processed ckpts (default: watch-dir/.lm_eval_submitted)
  --stop-file FILE     Stop when file exists (default: watch-dir/.training_complete)
EOF
}

TASKS="belebele"
BS="auto"
POLL=120
WANDB_PROJ="llama31_multilingual_eval_belebele"
WANDB_PREF="cola_moe_acc"
EXTRA=""
WATCH=""
SCRIPT=""
OUT=""
STATE=""
STOP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-dir) WATCH=$2;;
    --eval-script) SCRIPT=$2;;
    --tasks) TASKS=$2;;
    --batch-size) BS=$2;;
    --output-dir) OUT=$2;;
    --wandb-project) WANDB_PROJ=$2;;
    --wandb-prefix) WANDB_PREF=$2;;
    --extra-args) EXTRA=$2;;
    --poll-interval) POLL=$2;;
    --state-file) STATE=$2;;
    --stop-file) STOP=$2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option $1"; usage; exit 1;;
  esac
  shift
done

[[ -z "$WATCH" || -z "$SCRIPT" ]] && { echo "--watch-dir and --eval-script required"; exit 1; }

OUT=${OUT:-"$WATCH/lm_eval"}
STATE=${STATE:-"$WATCH/.lm_eval_submitted"}
STOP=${STOP:-"$WATCH/.training_complete"}

mkdir -p "$OUT" logs
touch "$STATE"

processed() { grep -Fxq "$1" "$STATE"; }
mark() { echo "$1" >> "$STATE"; }

submit() {
  echo "[INFO] eval for $1"
  sbatch "$SCRIPT" \
    --checkpoint "$1" \
    --output-dir "$OUT" \
    --tasks "$TASKS" \
    --batch-size "$BS" \
    --wandb-project "$WANDB_PROJ" \
    --wandb-prefix "$WANDB_PREF" \
    ${EXTRA:+--extra-args "$EXTRA"} \
  && mark "$1"
}

echo "[INFO] Watching $WATCH"

while true; do
  mapfile -t CKPTS < <(find "$WATCH" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

  for ckpt in "${CKPTS[@]}"; do
    processed "$ckpt" || submit "$ckpt"
  done

  if [[ -f "$STOP" ]]; then
    processed "$WATCH" || submit "$WATCH"
    echo "[INFO] Training done, exiting."
    exit 0
  fi

  sleep "$POLL"
done
