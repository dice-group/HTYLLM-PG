#!/bin/bash
# Launcher script for checkpoint listener - 3.7B multilingual model
#
# Run this AFTER starting the training job:
#   sbatch train_multilingual_3_7b.sh
#   bash start_eval_listener_3_7b.sh   # or sbatch this script
#
# The listener will watch for new checkpoints and submit evaluation jobs automatically.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Configuration for 3.7B model
WATCH_DIR="/scratch/hpc-prf-merlin/luke/checkpoints_multilingual"
EVAL_SCRIPT="${SCRIPT_DIR}/convert_and_eval.sh"
TASKS_FILE="${PROJECT_ROOT}/lm_eval_tasks.txt"
POLL_INTERVAL=300  # Check every 5 minutes
WANDB_PROJECT="htyllm-pg-eval"
WANDB_PREFIX="moe_3_7b"

echo "=========================================="
echo "Starting Checkpoint Listener for 3.7B Model"
echo "=========================================="
echo "Watch Directory: $WATCH_DIR"
echo "Eval Script:     $EVAL_SCRIPT"
echo "Tasks File:      $TASKS_FILE ($(wc -l < "$TASKS_FILE") tasks)"
echo "Poll Interval:   ${POLL_INTERVAL}s"
echo "W&B Project:     $WANDB_PROJECT"
echo "W&B Prefix:      $WANDB_PREFIX"
echo "=========================================="

# Submit the listener as a SLURM job (long-running)
sbatch "${SCRIPT_DIR}/checkpoint_listener.sh" \
  --watch-dir "$WATCH_DIR" \
  --eval-script "$EVAL_SCRIPT" \
  --poll-interval "$POLL_INTERVAL" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-prefix "$WANDB_PREFIX"

echo "Checkpoint listener submitted. Check logs/checkpoint_listener_*.log for status."
