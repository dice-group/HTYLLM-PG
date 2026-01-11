#!/bin/bash
#SBATCH --job-name=moe-diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

# Diagnostic test script for converted HuggingFace models
# Runs tests to identify issues with model conversion/inference
#
# Usage: sbatch run_model_diagnostics.sh --hf-model /path/to/hf_model [options]

set -euo pipefail

usage() {
cat <<EOF
Usage: run_model_diagnostics.sh --hf-model DIR [options]

Runs diagnostic tests on a converted HuggingFace model to identify
issues that could cause random-chance evaluation results.

Required:
  --hf-model DIR         Path to converted HuggingFace model directory

Options:
  --ds-checkpoint DIR    Path to original DeepSpeed checkpoint (for comparison tests)
  --config-path FILE     Model config JSON (default: auto-detect from HF model)
  --quick                Run only quick tests (weight sanity, logit range)
  --output-dir DIR       Where to save test results (default: diagnostics_results/)
  -h, --help             Show this help message
EOF
}

# ---------- Default values ----------
HF_MODEL=""
DS_CHECKPOINT=""
CONFIG_PATH=""
QUICK=false
OUTPUT_DIR="diagnostics_results"

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-model) HF_MODEL="$2"; shift 2;;
    --ds-checkpoint) DS_CHECKPOINT="$2"; shift 2;;
    --config-path) CONFIG_PATH="$2"; shift 2;;
    --quick) QUICK=true; shift;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

[[ -z "$HF_MODEL" ]] && { echo "ERROR: --hf-model is required"; usage; exit 1; }
[[ ! -d "$HF_MODEL" ]] && { echo "ERROR: HF model directory not found: $HF_MODEL"; exit 1; }

# ---------- Env ----------
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

# ---------- Resolve paths ----------
HF_MODEL=$(realpath "$HF_MODEL")
MODEL_NAME=$(basename "$HF_MODEL")

if [[ -n "$DS_CHECKPOINT" ]]; then
  DS_CHECKPOINT=$(realpath "$DS_CHECKPOINT")
fi

# Auto-detect config if not specified
if [[ -z "$CONFIG_PATH" ]]; then
  if [[ -f "${HF_MODEL}/config.json" ]]; then
    CONFIG_PATH="${HF_MODEL}/config.json"
  fi
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_diagnostics_${TIMESTAMP}.txt"

echo "=========================================="
echo "MoE Model Diagnostics"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "HF Model:     $HF_MODEL"
echo "DS Checkpoint: ${DS_CHECKPOINT:-<not set>}"
echo "Config:       ${CONFIG_PATH:-<auto>}"
echo "Quick Mode:   $QUICK"
echo "Output:       $RESULT_FILE"
echo "=========================================="

# ---------- Set environment variables for tests ----------
export HF_MODEL_PATH="$HF_MODEL"

if [[ -n "$DS_CHECKPOINT" ]]; then
  export DS_CHECKPOINT_PATH="$DS_CHECKPOINT"
fi

if [[ -n "$CONFIG_PATH" ]]; then
  export CONFIG_PATH="$CONFIG_PATH"
fi

# ---------- Run diagnostics ----------
echo "[INFO] Running diagnostic tests..."

# Build pytest arguments
PYTEST_ARGS="-v --tb=short"
if [[ "$QUICK" == true ]]; then
  PYTEST_ARGS="$PYTEST_ARGS -k 'weight_sanity or logits_range or vocab_size'"
fi

# Run tests and capture output
cd "$(dirname "$0")/../.."  # Go to project root

python -m pytest tests/test_converted_model.py $PYTEST_ARGS 2>&1 | tee "$RESULT_FILE"
TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "Diagnostics Complete!"
echo "=========================================="
echo "Exit Code: $TEST_EXIT_CODE"
echo "Results:   $RESULT_FILE"
echo "=========================================="

# ---------- Summary ----------
echo ""
echo "[SUMMARY]"
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
  echo "All tests PASSED - Model appears to be correctly converted."
else
  echo "Some tests FAILED - Check $RESULT_FILE for details."
  echo ""
  echo "Common issues to investigate:"
  echo "  1. Vocab size mismatch between tokenizer and model"
  echo "  2. Weight loading issues (NaN/Inf values)"
  echo "  3. MoE layers not routing tokens correctly"
  echo "  4. DeepSpeed initialization issues during inference"
fi

exit $TEST_EXIT_CODE
