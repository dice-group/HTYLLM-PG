#!/bin/bash
#SBATCH --job-name=moe-convert-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

# Combined conversion + evaluation script for DeepSpeed MoE checkpoints
# Called by checkpoint_listener.sh
#
# Usage: sbatch convert_and_eval.sh --checkpoint /path/to/checkpoint-2000 [options]

set -euo pipefail

usage() {
cat <<EOF
Usage: convert_and_eval.sh --checkpoint DIR [options]

Converts a DeepSpeed checkpoint to HuggingFace format and runs lm_eval.

Required:
  --checkpoint DIR       Path to DeepSpeed checkpoint directory (e.g., checkpoint-2000)

Options:
  --output-dir DIR       Where to save eval results (default: checkpoint parent/lm_eval)
  --tasks LIST           lm-eval tasks, comma-separated (overrides tasks file)
  --tasks-file FILE      File with tasks, one per line (default: lm_eval_tasks.txt)
  --batch-size N         lm-eval batch size (default: auto)
  --config-path FILE     Model config JSON (default: auto-detect based on checkpoint path)
  --hf-output-dir DIR    Where to save converted HF model (default: checkpoint parent/hf_models)
  --wandb-project NAME   W&B project name (default: htyllm-pg-eval)
  --wandb-prefix PREFIX  W&B run prefix (optional)
  --extra-args "ARGS"    Extra arguments to pass to lm_eval
  --skip-conversion      Skip conversion if HF model already exists
  --num-fewshot N        Number of few-shot examples (default: 0)
EOF
}

# ---------- Default values ----------
CHECKPOINT=""
OUTPUT_DIR=""
TASKS=""
TASKS_FILE="lm_eval_tasks.txt"
BS="auto"
CONFIG_PATH=""
HF_OUTPUT_DIR=""
WANDB_PROJ="htyllm-pg-eval"
WANDB_PREF=""
EXTRA_ARGS=""
SKIP_CONVERSION=false
NUM_FEWSHOT=0

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --tasks-file) TASKS_FILE="$2"; shift 2;;
    --batch-size) BS="$2"; shift 2;;
    --config-path) CONFIG_PATH="$2"; shift 2;;
    --hf-output-dir) HF_OUTPUT_DIR="$2"; shift 2;;
    --wandb-project) WANDB_PROJ="$2"; shift 2;;
    --wandb-prefix) WANDB_PREF="$2"; shift 2;;
    --extra-args) EXTRA_ARGS="$2"; shift 2;;
    --skip-conversion) SKIP_CONVERSION=true; shift;;
    --num-fewshot) NUM_FEWSHOT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

[[ -z "$CHECKPOINT" ]] && { echo "ERROR: --checkpoint is required"; usage; exit 1; }
[[ ! -d "$CHECKPOINT" ]] && { echo "ERROR: Checkpoint directory not found: $CHECKPOINT"; exit 1; }

# ---------- Load tasks from file if not specified via --tasks ----------
if [[ -z "$TASKS" ]]; then
  if [[ -f "$TASKS_FILE" ]]; then
    echo "[INFO] Loading tasks from $TASKS_FILE"
    # Read file, skip empty lines and comments, join with commas
    TASKS=$(grep -v '^#' "$TASKS_FILE" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
    echo "[INFO] Loaded $(grep -v '^#' "$TASKS_FILE" | grep -v '^$' | wc -l) tasks"
  else
    echo "ERROR: No --tasks specified and tasks file not found: $TASKS_FILE"
    exit 1
  fi
fi

# ---------- Env ----------
# Temporarily disable strict mode for bashrc/conda/modules
# (they use unbound variables internally and may source /etc/bashrc)
set +euo pipefail
source ~/.bashrc 2>/dev/null || true
conda activate moe 2>/dev/null || true
module load system/CUDA/12.6.0 2>/dev/null || true
module load compiler/GCCcore/12.3.0 2>/dev/null || true
set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p /scratch/hpc-prf-merlin/luke/.cache/torch_extensions
mkdir -p /scratch/hpc-prf-merlin/luke/.triton/autotune
export TORCH_EXTENSIONS_DIR="/scratch/hpc-prf-merlin/luke/.cache/torch_extensions"
export TRITON_CACHE_DIR="/scratch/hpc-prf-merlin/luke/.triton/autotune"
export XDG_CACHE_HOME="/scratch/hpc-prf-merlin/luke/.cache"
export CUDA_LAUNCH_BLOCKING=1

# ---------- Derive paths ----------
CHECKPOINT_PARENT=$(dirname "$CHECKPOINT")
CHECKPOINT_NAME=$(basename "$CHECKPOINT")

# Auto-detect config based on checkpoint path
if [[ -z "$CONFIG_PATH" ]]; then
  if [[ "$CHECKPOINT_PARENT" == *"multilingual_small"* ]]; then
    CONFIG_PATH="htyllm_pg/conversion_scripts/config_small.json"
  else
    CONFIG_PATH="htyllm_pg/conversion_scripts/config_3_7b.json"
  fi
fi

HF_OUTPUT_DIR=${HF_OUTPUT_DIR:-"${CHECKPOINT_PARENT}/hf_models"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CHECKPOINT_PARENT}/lm_eval"}
HF_MODEL_PATH="${HF_OUTPUT_DIR}/${CHECKPOINT_NAME}"

mkdir -p "$HF_OUTPUT_DIR" "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="${OUTPUT_DIR}/${CHECKPOINT_NAME}_${TIMESTAMP}.json"

# Count number of tasks
NUM_TASKS=$(echo "$TASKS" | tr ',' '\n' | wc -l)

echo "=========================================="
echo "MoE Convert + Evaluate Pipeline"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Checkpoint:   $CHECKPOINT"
echo "Config:       $CONFIG_PATH"
echo "HF Output:    $HF_MODEL_PATH"
echo "Eval Output:  $RESULT_FILE"
echo "Tasks:        $NUM_TASKS tasks loaded"
echo "Batch Size:   $BS"
echo "W&B Project:  $WANDB_PROJ"
echo "W&B Prefix:   ${WANDB_PREF:-<none>}"
echo "=========================================="

# ---------- Step 1: Convert DeepSpeed → HuggingFace ----------
if [[ -f "${HF_MODEL_PATH}/config.json" ]] && [[ "$SKIP_CONVERSION" == true ]]; then
  echo "[INFO] HF model already exists at ${HF_MODEL_PATH}, skipping conversion."
else
  echo "[INFO] Converting DeepSpeed checkpoint to HuggingFace format..."
  
  # Check if config exists
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
  fi
  
  # Run conversion with deepspeed (single GPU)
  deepspeed --num_gpus=1 htyllm_pg/conversion_scripts/convert_ds_to_hf.py \
    --checkpoint_path "$CHECKPOINT" \
    --output_dir "$HF_MODEL_PATH" \
    --config_path "$CONFIG_PATH"
  
  echo "[INFO] Conversion complete: $HF_MODEL_PATH"
fi

# Verify conversion succeeded
if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
  echo "ERROR: Conversion failed - no config.json found in $HF_MODEL_PATH"
  exit 1
fi

# ---------- Step 2: Run lm_eval ----------
echo "[INFO] Running lm_eval..."

# Build lm_eval command
LM_EVAL_CMD="lm_eval \
  --model hf \
  --model_args pretrained=${HF_MODEL_PATH},trust_remote_code=True \
  --tasks ${TASKS} \
  --batch_size ${BS} \
  --num_fewshot ${NUM_FEWSHOT} \
  --output_path ${RESULT_FILE}"

# Add extra args if provided
if [[ -n "$EXTRA_ARGS" ]]; then
  LM_EVAL_CMD="$LM_EVAL_CMD $EXTRA_ARGS"
fi

echo "Command: $LM_EVAL_CMD"
eval "$LM_EVAL_CMD"

echo "=========================================="
echo "Evaluation Complete!"
echo "Results: $RESULT_FILE"
echo "=========================================="

# ---------- Step 3: Log results to W&B ----------
echo "[INFO] Logging results to Weights & Biases..."

# Determine run name
if [[ -n "$WANDB_PREF" ]]; then
  WANDB_RUN_NAME="${WANDB_PREF}_${CHECKPOINT_NAME}"
else
  WANDB_RUN_NAME="eval_${CHECKPOINT_NAME}"
fi

# Determine model variant from checkpoint path for grouping
if [[ "$CHECKPOINT_PARENT" == *"multilingual_small"* ]]; then
  MODEL_VARIANT="small"
else
  MODEL_VARIANT="3_7b"
fi

python3 << PYTHON_SCRIPT
import json
import wandb
import os
import re

# W&B API key (same as train.py)
wandb.login(key="844fd819fc05b9e11ac9814b166ab940a5579dfb")

# Extract step number from checkpoint name (e.g., "step_2000" -> 2000)
checkpoint_name = "${CHECKPOINT_NAME}"
step_match = re.search(r'step_(\d+)', checkpoint_name)
step = int(step_match.group(1)) if step_match else 0

# Load results
result_path = "${RESULT_FILE}"

# lm_eval creates a nested directory structure. Recursively find the results JSON.
def find_results_json(path):
    """Recursively find results*.json file in lm_eval output directory."""
    if os.path.isfile(path) and path.endswith('.json'):
        return path
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.startswith('results') and f.endswith('.json'):
                    return os.path.join(root, f)
        # Fallback: any .json file
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.json'):
                    return os.path.join(root, f)
    return None

result_file = find_results_json(result_path)
if result_file is None:
    print(f"No results JSON found in: {result_path}")
    exit(1)

print(f"Found results file: {result_file}")

try:
    with open(result_file, 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"Results file not found: {result_file}")
    exit(1)

# Load model config for logging
config_path = "${CONFIG_PATH}"
model_config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        model_config = json.load(f)

# Derive group name from checkpoint parent directory for line plots
# e.g., "/scratch/.../checkpoints_multilingual" -> "checkpoints_multilingual"
checkpoint_parent = "${CHECKPOINT_PARENT}"
group_name = os.path.basename(checkpoint_parent)

# Create a consistent run ID for resuming (allows line plot across checkpoints)
# Format: {group_name}_eval_v2 (v2 = new metric structure with task-level sections)
run_id = f"{group_name}_eval_v2".replace("/", "_").replace(" ", "_")

# Initialize W&B run in OFFLINE mode (HPC compute nodes have restricted network)
# Sync later from login node with: wandb sync wandb/offline-run-*
run = wandb.init(
    project="${WANDB_PROJ}",
    id=run_id,
    name=f"eval_{group_name}",
    resume="allow",  # Resume if exists, create if not
    config={
        "model_variant": "${MODEL_VARIANT}",
        "tasks": "${TASKS}",
        "num_fewshot": ${NUM_FEWSHOT},
        "batch_size": "${BS}",
        "checkpoint_dir": checkpoint_parent,
        **model_config
    },
    tags=["eval", "${MODEL_VARIANT}"],
    mode="offline"  # Save locally, sync later from login node
)

# Define step metric for proper x-axis (line plot!)
wandb.define_metric("step")

# Parse and log results
log_dict = {"step": step}

print(f"Logging eval results for step {step}...")
print("Task Results:")

# Collect all unique task names for metric definitions
task_names = set()
for task, metrics in results.get('results', {}).items():
    task_names.add(task)
    print(f"  {task}:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            # Clean metric name for wandb (remove special chars)
            clean_metric = metric.replace(',', '_').replace(' ', '_')
            # Use task as top-level section (e.g., "hellaswag/acc" instead of "eval/hellaswag/acc")
            log_dict[f"{task}/{clean_metric}"] = value
            print(f"    {metric}: {value:.4f}")

# Define metrics per task section so they use step as x-axis
for task in task_names:
    wandb.define_metric(f"{task}/*", step_metric="step")

wandb.log(log_dict)

# Log the full results JSON as an artifact (versioned per checkpoint)
artifact = wandb.Artifact(
    name=f"eval_results_{group_name}",
    type="eval_results",
    description=f"lm_eval results for {checkpoint_name}"
)
# Add the actual results file (not the directory path)
artifact.add_file(result_file)
run.log_artifact(artifact)

wandb.finish()
print(f"Results logged to W&B project: ${WANDB_PROJ}")
print(f"Run ID: {run_id} (resume=allow creates line plot over steps)")
PYTHON_SCRIPT
