#!/usr/bin/env bash
set -euo pipefail

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

TASKS=${TASKS:-belebele}
BS=${BS:-auto}
TOK=${TOK:-$CKPT}
WP=${WP:-lm_eval_debug}
PREF=${PREF:-lm_eval}
WGROUP=${WGROUP:-}
WJOB=${WJOB:-checkpoint_eval}
WRESUME=${WRESUME:-allow}
WMODE=${WMODE:-shared}
EXTRA=${EXTRA:-}

[[ -z "${CKPT:-}" || -z "${OUTDIR:-}" ]] && { echo "--checkpoint and --output-dir required"; exit 1; }
[[ ! -d "${CKPT}" ]] && { echo "Checkpoint not found: ${CKPT}"; exit 1; }

if [[ ! -f "${CKPT}/adapter_config.json" && -f "${CKPT}_adapter/adapter_config.json" ]]; then
  CKPT="${CKPT}_adapter"
fi

MODEL_ARGS="pretrained=${CKPT},tokenizer=${TOK}"
if [[ -f "${CKPT}/adapter_config.json" ]]; then
  BASE=${TOK}
  if [[ -z "${BASE}" || "${BASE}" == "${CKPT}" ]]; then
    BASE=$(python3 - <<'PY' "${CKPT}/adapter_config.json"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("base_model_name_or_path", ""))
PY
)
  fi
  [[ -z "${BASE}" ]] && { echo "Base model not found for adapter checkpoint: ${CKPT}"; exit 1; }
  TOK_USE=${TOK}
  if [[ -z "${TOK_USE}" || "${TOK_USE}" == "${CKPT}" ]]; then
    TOK_USE=${BASE}
  fi
  MODEL_ARGS="pretrained=${BASE},peft=${CKPT},tokenizer=${TOK_USE}"
fi

mkdir -p "${OUTDIR}"

LABEL=$(basename "${CKPT}")
OUTFILE="${OUTDIR}/${LABEL}_lm_eval.jsonl"
WANDB_NAME="${PREF}_${LABEL}"
WANDB_ARGS="project=${WP},name=${WANDB_NAME}"
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

LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

"${LM_EVAL_BIN}" \
  --model hf \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size "${BS}" \
  --output_path "${OUTFILE}" \
  --wandb_args "${WANDB_ARGS}" \
  ${EXTRA}
