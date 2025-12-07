#!/bin/bash
#SBATCH --job-name=moe-ckpt-listener
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=36:30:00
#SBATCH --output=logs/checkpoint_listener_%j.log                #Check this path?

# This script can run either via `sbatch checkpoint_listener.sh ...`
# (using the header above) or directly with `bash checkpoint_listener.sh ...`.
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
WANDB_PROJ=""            #"llama31_multilingual_eval_belebele"            #Insert our WANDB
WANDB_PREF=""            #"cola_moe_acc"
EXTRA=""
WATCH=""
SCRIPT=""
OUT=""
STATE=""
STOP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-dir) WATCH=$2; shift 2; continue;;
    --eval-script) SCRIPT=$2; shift 2; continue;;
    --tasks) TASKS=$2; shift 2; continue;;
    --batch-size) BS=$2; shift 2; continue;;
    --output-dir) OUT=$2; shift 2; continue;;
    --wandb-project) WANDB_PROJ=$2; shift 2; continue;;
    --wandb-prefix) WANDB_PREF=$2; shift 2; continue;;
    --extra-args) EXTRA=$2; shift 2; continue;;
    --poll-interval) POLL=$2; shift 2; continue;;
    --state-file) STATE=$2; shift 2; continue;;
    --stop-file) STOP=$2; shift 2; continue;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option $1"; usage; exit 1;;
  esac
done

[[ -z "$WATCH" || -z "$SCRIPT" ]] && { echo "--watch-dir and --eval-script required"; exit 1; }

mkdir -p "$WATCH"
WATCH_LABEL=$(basename "$WATCH")

OUT=${OUT:-"$WATCH/lm_eval"}
STATE=${STATE:-"$WATCH/.lm_eval_submitted"}
STOP=${STOP:-"$WATCH/.training_complete"}

mkdir -p "$OUT" logs
touch "$STATE"

processed() { grep -Fxq "$1" "$STATE"; }
mark() { echo "$1" >> "$STATE"; }

submit() {
  local ckpt_path=$1
  echo "[INFO] eval for ${ckpt_path}"
  local ckpt_label
  ckpt_label=$(basename "${ckpt_path}")
  local job_name="lm-eval_${WATCH_LABEL}_${ckpt_label}"
  local job_log="logs/${job_name}_%j.log"
  local wandb_prefix="${WANDB_PREF}_${WATCH_LABEL}"
  sbatch \
    --job-name="${job_name}" \
    --output="${job_log}" \
    "$SCRIPT" \
    --checkpoint "${ckpt_path}" \
    --output-dir "$OUT" \
    --tasks "$TASKS" \
    --batch-size "$BS" \
    --wandb-project "$WANDB_PROJ" \
    --wandb-prefix "${wandb_prefix}" \
    ${EXTRA:+--extra-args "$EXTRA"} \
  && mark "${ckpt_path}"
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