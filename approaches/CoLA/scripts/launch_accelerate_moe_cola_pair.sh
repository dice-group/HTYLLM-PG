#!/bin/bash

# Helper script to queue two CoLA runs:
#   1) Baseline PiSSA init using defaults from accelerate_moe_cola_train.sh
#   2) Non-PiSSA init with NUM_A=NUM_B=1 (still 4 experts, top-k=1)

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT="${SCRIPT_DIR}/accelerate_moe_cola_train.sh"
LM_EVAL_SCRIPT="${SCRIPT_DIR}/lm_eval_checkpoint.sh"
CHECKPOINT_LISTENER_SCRIPT="${SCRIPT_DIR}/checkpoint_listener.sh"
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/saves/cola_moe_llama31_8b_acc}
PISSA_OUTPUT_DIR="${PISSA_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/pissa}"
NOPISSA_OUTPUT_DIR="${NOPISSA_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/nopissa}"
ENABLE_CHECKPOINT_LISTENER=${ENABLE_CHECKPOINT_LISTENER:-1}
LM_EVAL_TASKS=${LM_EVAL_TASKS:-belebele}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-auto}
LM_EVAL_WANDB_PROJECT=${LM_EVAL_WANDB_PROJECT:-llama31_multilingual_eval_belebele}
LM_EVAL_POLL_INTERVAL=${LM_EVAL_POLL_INTERVAL:-120}
LM_EVAL_EXTRA_ARGS=${LM_EVAL_EXTRA_ARGS:-}
PISSA_WANDB_PREFIX=${PISSA_WANDB_PREFIX:-cola_moe_acc_pissa}
NOPISSA_WANDB_PREFIX=${NOPISSA_WANDB_PREFIX:-cola_moe_acc_nopissa}

export LM_EVAL_TASKS LM_EVAL_BATCH_SIZE LM_EVAL_WANDB_PROJECT LM_EVAL_POLL_INTERVAL LM_EVAL_EXTRA_ARGS

mkdir -p logs

launch_listener_job() {
  local watch_dir=$1
  local wandb_prefix=$2
  local label=$3

  if [[ "${ENABLE_CHECKPOINT_LISTENER}" != "1" ]]; then
    return
  fi
  if [[ ! -x "${CHECKPOINT_LISTENER_SCRIPT}" ]]; then
    echo "[WARN] Checkpoint listener script missing or not executable at ${CHECKPOINT_LISTENER_SCRIPT}; skipping listener launch for ${watch_dir}." >&2
    return
  fi
  if [[ ! -f "${LM_EVAL_SCRIPT}" ]]; then
    echo "[WARN] LM eval script not found at ${LM_EVAL_SCRIPT}; skipping listener launch for ${watch_dir}." >&2
    return
  fi

  local listener_args=(
    --watch-dir "${watch_dir}"
    --eval-script "${LM_EVAL_SCRIPT}"
    --tasks "${LM_EVAL_TASKS}"
    --batch-size "${LM_EVAL_BATCH_SIZE}"
    --wandb-project "${LM_EVAL_WANDB_PROJECT}"
    --wandb-prefix "${wandb_prefix}"
    --poll-interval "${LM_EVAL_POLL_INTERVAL}"
  )
  if [[ -n "${LM_EVAL_EXTRA_ARGS}" ]]; then
    listener_args+=(--extra-args "${LM_EVAL_EXTRA_ARGS}")
  fi

  local listener_job_name="${label}-listener"
  local listener_log="logs/${listener_job_name}_%j.log"
  local submit_output
  submit_output=$(
    sbatch \
      --job-name="${listener_job_name}" \
      --output="${listener_log}" \
      "${CHECKPOINT_LISTENER_SCRIPT}" \
      "${listener_args[@]}"
  )
  local listener_job_id
  listener_job_id=$(echo "${submit_output}" | awk '{print $4}')
  if [[ -n "${listener_job_id}" ]]; then
    echo "[INFO] Submitted checkpoint listener job ${listener_job_id} for ${watch_dir}"
  else
    echo "[WARN] Unable to determine listener job ID from submission output: ${submit_output}" >&2
  fi
}

# PiSSA configuration
export USE_COLA_PISSA_INIT=True
export COLA_INIT_LORA_WEIGHTS=
export NUM_A=2
export NUM_B=4
export OUTPUT_DIR="${PISSA_OUTPUT_DIR}"
export LM_EVAL_WANDB_PREFIX="${PISSA_WANDB_PREFIX}"

echo "[INFO] Submitting PiSSA-enabled CoLA run (defaults) to ${OUTPUT_DIR}."
PISSA_JOB_ID=$(
  sbatch \
    --job-name=cola-moe-pissa \
    --output=logs/train_acc_pissa_%j.log \
    "${TRAIN_SCRIPT}" \
    | awk '{print $4}'
)
echo "[INFO] Submitted PiSSA job ${PISSA_JOB_ID}"
launch_listener_job "${PISSA_OUTPUT_DIR}" "${PISSA_WANDB_PREFIX}" "cola-moe-pissa"

# non-PiSSA configuration
export USE_COLA_PISSA_INIT=False
export COLA_INIT_LORA_WEIGHTS=
export NUM_A=1
export NUM_B=1
export OUTPUT_DIR="${NOPISSA_OUTPUT_DIR}"
export LM_EVAL_WANDB_PREFIX="${NOPISSA_WANDB_PREFIX}"

echo "[INFO] Submitting non-PiSSA CoLA run (NUM_A=NUM_B=1) to ${OUTPUT_DIR}."
NOPISSA_JOB_ID=$(
  sbatch \
    --job-name=cola-moe-nopissa \
    --output=logs/train_acc_nopissa_%j.log \
    "${TRAIN_SCRIPT}" \
    | awk '{print $4}'
)
echo "[INFO] Submitted non-PiSSA job ${NOPISSA_JOB_ID}"
launch_listener_job "${NOPISSA_OUTPUT_DIR}" "${NOPISSA_WANDB_PREFIX}" "cola-moe-nopissa"

echo "[INFO] Both runs queued. Use 'squeue -u $USER' to monitor progress."
