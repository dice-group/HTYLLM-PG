#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<EOF
Usage: checkpoint_listener_local.sh --watch-dir DIR --eval-script SCRIPT [options]

Scans checkpoints in a directory and runs lm-eval locally for each one.

Options:
  --watch-dir DIR      Directory containing checkpoint-* folders
  --eval-script SCRIPT Script run locally for evaluation
  --tokenizer PATH     Tokenizer to pass to eval script (defaults to checkpoint)
  --tasks LIST         lm-eval tasks (default: belebele)
  --batch-size N       lm-eval batch size (default: auto)
  --output-dir DIR     Where to save eval outputs (default: watch-dir/lm_eval)
  --wandb-project NAME W&B project (default: lm_eval_debug)
  --wandb-prefix PREF  W&B prefix (default: lm_eval)
  --wandb-group NAME   W&B group to tie checkpoint runs together (default: watch-dir basename)
  --wandb-id FILE|ID   W&B run id or file containing the id (default: watch-dir/.wandb_eval_run_id)
  --wandb-resume MODE  W&B resume mode (default: allow)
  --wandb-mode MODE    W&B mode (default: shared)
  --extra-args "ARGS"  Extra args for lm_eval
  --state-file FILE    Track processed ckpts (default: watch-dir/.lm_eval_submitted)
  --stop-file FILE     Stop when file exists (default: watch-dir/.training_complete)
  --once               Process existing checkpoints once and exit
EOF
}

TASKS="belebele"
BS="auto"
TOK=""
WANDB_PROJ="lm_eval_debug"
WANDB_PREF="lm_eval"
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
ONCE="false"

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
    --state-file) STATE=$2; shift 2; continue;;
    --stop-file) STOP=$2; shift 2; continue;;
    --once) ONCE="true"; shift 1; continue;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option $1"; usage; exit 1;;
  esac
done

[[ -z "${WATCH}" || -z "${SCRIPT}" ]] && { echo "--watch-dir and --eval-script required"; exit 1; }

mkdir -p "${WATCH}"
WATCH_LABEL=$(basename "${WATCH}")

OUT=${OUT:-"${WATCH}/lm_eval"}
STATE=${STATE:-"${WATCH}/.lm_eval_submitted"}
STOP=${STOP:-"${WATCH}/.training_complete"}
WANDB_GROUP=${WANDB_GROUP:-"${WATCH_LABEL}"}
WANDB_ID=${WANDB_ID:-"${WATCH}/.wandb_eval_run_id"}

resolve_wandb_id() {
  local id_spec=$1
  if [[ -f "${id_spec}" ]]; then
    local existing
    existing=$(tr -d '[:space:]' < "${id_spec}")
    if [[ -n "${existing}" ]]; then
      echo "${existing}"
      return 0
    fi
  fi
  if [[ "${id_spec}" == */* || "${id_spec}" == *.* ]]; then
    local new_id
    new_id=$(date +%s%N)
    new_id="eval_${WATCH_LABEL}_${new_id}"
    echo "${new_id}" > "${id_spec}"
    echo "${new_id}"
    return 0
  fi
  echo "${id_spec}"
}

WANDB_RUN_ID=$(resolve_wandb_id "${WANDB_ID}")

mkdir -p "${OUT}"
touch "${STATE}"

processed() { grep -Fxq "$1" "${STATE}"; }
mark() { echo "$1" >> "${STATE}"; }

run_eval() {
  local ckpt_path=$1
  "${SCRIPT}" \
    --checkpoint "${ckpt_path}" \
    --output-dir "${OUT}" \
    --tasks "${TASKS}" \
    --batch-size "${BS}" \
    --wandb-project "${WANDB_PROJ}" \
    --wandb-prefix "${WANDB_PREF}_${WATCH_LABEL}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-id "${WANDB_RUN_ID}" \
    --wandb-resume "${WANDB_RESUME}" \
    --wandb-mode "${WANDB_MODE}" \
    ${TOK:+--tokenizer "${TOK}"} \
    ${EXTRA:+--extra-args "${EXTRA}"} \
  && mark "${ckpt_path}"
}

scan_once() {
  mapfile -t CKPTS < <(find "${WATCH}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  for ckpt in "${CKPTS[@]}"; do
    processed "${ckpt}" || run_eval "${ckpt}"
  done
  if [[ -f "${STOP}" ]]; then
    processed "${WATCH}" || run_eval "${WATCH}"
  fi
}

scan_once
if [[ "${ONCE}" == "true" ]]; then
  exit 0
fi

while true; do
  scan_once
  sleep 60
done
