#!/bin/bash
#SBATCH --job-name=cola-lm-eval
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/lm_eval_%x_%j.log

set -e

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)   CKPT=$2; shift 2;;
    --tokenizer)    TOK=$2; shift 2;;
    --tasks)        TASKS=$2; shift 2;;
    --batch-size)   BS=$2; shift 2;;
    --output-dir)   OUTDIR=$2; shift 2;;
    --wandb-project) WP=$2; shift 2;;
    --wandb-prefix)  PREF=$2; shift 2;;
    --wandb-group)   WGROUP=$2; shift 2;;
    --wandb-id)      WID=$2; shift 2;;
    --wandb-resume)  WRESUME=$2; shift 2;;
    --wandb-mode)    WMODE=$2; shift 2;;
    --wandb-job-type) WJOB=$2; shift 2;;
    --extra-args)    EXTRA=$2; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

# params
TASKS=${TASKS:-belebele}
BS=${BS:-auto}
TOK=${TOK:-$CKPT}
WP=${WP:-llama31_multilingual_eval_belebele}
PREF=${PREF:-cola_moe_acc}
WGROUP=${WGROUP:-}
WJOB=${WJOB:-checkpoint_eval}
WRESUME=${WRESUME:-allow}
WMODE=${WMODE:-shared}

[[ -z "$CKPT" || -z "$OUTDIR" ]] && { echo "--checkpoint and --output-dir required"; exit 1; }
[[ ! -d "$CKPT" ]] && { echo "Checkpoint not found: $CKPT"; exit 1; }

MODEL_ARGS="pretrained=$CKPT,tokenizer=$TOK"
if [[ -f "$CKPT/adapter_config.json" ]]; then
  BASE=${TOK}
  if [[ -z "$BASE" || "$BASE" == "$CKPT" ]]; then
    BASE=$(python - <<'PY' "$CKPT/adapter_config.json"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("base_model_name_or_path", ""))
PY
)
  fi
  [[ -z "$BASE" ]] && { echo "Base model not found for adapter checkpoint: $CKPT"; exit 1; }
  TOK_USE=$TOK
  if [[ -z "$TOK_USE" || "$TOK_USE" == "$CKPT" ]]; then
    TOK_USE=$BASE
  fi
  MODEL_ARGS="pretrained=$BASE,peft=$CKPT,tokenizer=$TOK_USE"
fi

mkdir -p "$OUTDIR" logs
module purge
module load toolchain/foss/2024a system/CUDA/12.6.0 lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate cola_llama_factory

export PYTHONUNBUFFERED=1

# run params
LABEL=$(basename "$CKPT")
OUTFILE="${OUTDIR}/${LABEL}_lm_eval.jsonl"
WANDB_NAME="${PREF}_${LABEL}"
WANDB_ARGS="project=$WP,name=$WANDB_NAME"
if [[ -n "${WGROUP}" ]]; then
  WANDB_ARGS="${WANDB_ARGS},group=${WGROUP}"
fi
if [[ -n "${WID}" ]]; then
  WANDB_ARGS="${WANDB_ARGS},id=${WID}"
fi
if [[ -n "${WRESUME}" ]]; then
  WANDB_ARGS="${WANDB_ARGS},resume=${WRESUME}"
fi
if [[ -n "${WMODE}" ]]; then
  WANDB_ARGS="${WANDB_ARGS},mode=${WMODE}"
fi
if [[ -n "${WJOB}" ]]; then
  WANDB_ARGS="${WANDB_ARGS},job_type=${WJOB}"
fi

echo "Running lm-eval on $LABEL..."
lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BS" \
  --output_path "$OUTFILE" \
  --wandb_args "$WANDB_ARGS" \
  $EXTRA

echo "Done!"
