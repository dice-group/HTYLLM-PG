#!/bin/bash
#SBATCH --job-name=cola-ckpt-listener
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=168:00:00
#SBATCH --output=logs/checkpoint_listener_%j.log

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
  --tokenizer PATH     Tokenizer to pass to eval script (defaults to checkpoint)
  --tasks LIST         lm-eval tasks (default: belebele)
  --batch-size N       lm-eval batch size (default: auto)
  --output-dir DIR     Where to save eval outputs (default: watch-dir/lm_eval)
  --wandb-project NAME W&B project (default: llama31_multilingual_eval_belebele)
  --wandb-prefix PREF  W&B prefix (default: cola_moe_acc)
  --wandb-group NAME   W&B group to tie checkpoint runs together (default: watch-dir basename)
  --wandb-id FILE|ID   W&B run id or file containing the id (default: watch-dir/.wandb_eval_run_id)
  --wandb-resume MODE  W&B resume mode (default: allow)
  --wandb-mode MODE    W&B mode (default: shared)
  --extra-args "ARGS"  Extra args for lm_eval
  --poll-interval SEC  Scan interval (default: 120)
  --state-file FILE    Track processed ckpts (default: watch-dir/.lm_eval_submitted)
  --stop-file FILE     Stop when file exists (default: watch-dir/.training_complete)
EOF
}

TASKS="belebele"
BS="auto"
TOK=""
POLL=120
WANDB_PROJ="llama31_multilingual_eval_belebele"
WANDB_PREF="cola_moe_acc"
WANDB_GROUP=""
WANDB_ID=""
WANDB_RESUME="allow"
WANDB_MODE="shared"
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
    --tokenizer) TOK=$2; shift 2; continue;;
    --batch-size) BS=$2; shift 2; continue;;
    --output-dir) OUT=$2; shift 2; continue;;
    --wandb-project) WANDB_PROJ=$2; shift 2; continue;;
    --wandb-prefix) WANDB_PREF=$2; shift 2; continue;;
    --wandb-group) WANDB_GROUP=$2; shift 2; continue;;
    --wandb-id) WANDB_ID=$2; shift 2; continue;;
    --wandb-resume) WANDB_RESUME=$2; shift 2; continue;;
    --wandb-mode) WANDB_MODE=$2; shift 2; continue;;
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
WANDB_GROUP=${WANDB_GROUP:-"$WATCH_LABEL"}
WANDB_ID=${WANDB_ID:-"$WATCH/.wandb_eval_run_id"}

resolve_wandb_id() {
  local id_spec=$1
  if [[ -f "$id_spec" ]]; then
    local existing
    existing=$(cat "$id_spec" | tr -d '[:space:]')
    if [[ -n "$existing" ]]; then
      echo "$existing"
      return 0
    fi
  fi
  if [[ "$id_spec" == */* || "$id_spec" == *.* ]]; then
    local new_id
    new_id=$(date +%s%N)
    new_id="eval_${WATCH_LABEL}_${new_id}"
    echo "$new_id" > "$id_spec"
    echo "$new_id"
    return 0
  fi
  echo "$id_spec"
}

WANDB_RUN_ID=$(resolve_wandb_id "$WANDB_ID")

mkdir -p "$OUT" logs
touch "$STATE"

processed() { grep -Fxq "$1" "$STATE"; }
mark() { echo "$1" >> "$STATE"; }

resolve_eval_target() {
  local ckpt_path=$1
  local adapter_path="${ckpt_path}_adapter"
  local adapter_sharded="${ckpt_path}_adapter_sharded"
  if [[ -f "${ckpt_path}/adapter_config.json" ]]; then
    if [[ -f "${ckpt_path}/adapter_model.safetensors" || -f "${ckpt_path}/adapter_model.bin" ]]; then
      echo "${ckpt_path}"
      return
    fi
  fi
  if [[ -f "${adapter_path}/adapter_config.json" ]]; then
    if [[ -f "${adapter_path}/adapter_model.safetensors" || -f "${adapter_path}/adapter_model.bin" ]]; then
      echo "${adapter_path}"
      return
    fi
  fi
  if [[ -d "${adapter_sharded}" ]]; then
    echo "${ckpt_path}"
    return
  fi
  if [[ -f "${ckpt_path}/adapter_config.json" || -d "${adapter_path}" || -d "${adapter_sharded}" ]]; then
    echo ""
    return
  fi
  echo ""
}

submit() {
  local ckpt_path=$1
  echo "[INFO] eval for ${ckpt_path}"
  local ckpt_label
  ckpt_label=$(basename "${ckpt_path}")
  local job_name="lm-eval_${WATCH_LABEL}_${ckpt_label}"
  local job_log="${OUT}/logs/${job_name}_%j.log"
  local wandb_prefix="${WANDB_PREF}_${WATCH_LABEL}"
  mkdir -p "${OUT}/logs"
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
    --wandb-group "${WANDB_GROUP}" \
    --wandb-id "${WANDB_RUN_ID}" \
    --wandb-resume "${WANDB_RESUME}" \
    --wandb-mode "${WANDB_MODE}" \
    ${TOK:+--tokenizer "$TOK"} \
    ${EXTRA:+--extra-args "$EXTRA"} \
  && mark "${ckpt_path}"
}

echo "[INFO] Watching $WATCH"

while true; do
  mapfile -t CKPTS < <(find "$WATCH" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

  for ckpt in "${CKPTS[@]}"; do
    eval_target=$(resolve_eval_target "$ckpt")
    [[ -z "$eval_target" ]] && continue
    processed "$eval_target" || submit "$eval_target"
  done

  if [[ -f "$STOP" ]]; then
    if [[ ${#CKPTS[@]} -gt 0 ]]; then
      last_ckpt="${CKPTS[-1]}"
      eval_target=$(resolve_eval_target "$last_ckpt")
      if [[ -n "$eval_target" ]]; then
        processed "$eval_target" || submit "$eval_target"
      fi
    else
      processed "$WATCH" || submit "$WATCH"
    fi
    echo "[INFO] Training done, exiting."
    exit 0
  fi

  sleep "$POLL"
done
