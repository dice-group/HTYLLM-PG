#!/bin/bash
# Launcher script for checkpoint listener - Small multilingual model
#
# Run this AFTER starting the training job:
#   sbatch train_multilingual_small.sh
#   bash start_eval_listener_small.sh   # or sbatch this script
#
# The listener will watch for new checkpoints and submit evaluation jobs automatically.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Configuration for small model
WATCH_DIR="/scratch/hpc-prf-merlin/luke/checkpoints_multilingual_small"
EVAL_SCRIPT="${SCRIPT_DIR}/convert_and_eval.sh"
TASKS_FILE="${PROJECT_ROOT}/lm_eval_tasks.txt"
POLL_INTERVAL=180  # Check every 3 minutes (smaller model trains faster)
WANDB_PROJECT="htyllm-pg-eval"
WANDB_PREFIX="moe_small"

echo "=========================================="
echo "Starting Checkpoint Listener for Small Model"
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
