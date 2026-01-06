#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
    --lang-mode)     LANG_MODE=$2; shift 2;;
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
LANG_MODE=${LANG_MODE:-${LM_EVAL_LANG_MODE:-both}}
USE_LANG_WRAPPER=${LM_EVAL_USE_LANG_WRAPPER:-auto}
LOG_ROUTER_METRICS=${LM_EVAL_LOG_ROUTER_METRICS:-true}
LIMIT=${LM_EVAL_LIMIT:-}

[[ -z "${CKPT:-}" || -z "${OUTDIR:-}" ]] && { echo "--checkpoint and --output-dir required"; exit 1; }
[[ ! -d "${CKPT}" ]] && { echo "Checkpoint not found: ${CKPT}"; exit 1; }

ORIG_CKPT="${CKPT}"
if [[ ! -f "${CKPT}/adapter_config.json" && -d "${CKPT}_adapter_sharded" ]]; then
  base_model=""
  if [[ -f "${CKPT}_adapter_sharded/adapter_config.json" ]]; then
    base_model=$(python3 - <<'PY' "${CKPT}_adapter_sharded/adapter_config.json"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("base_model_name_or_path", ""))
PY
)
  fi
  if [[ -z "${base_model}" ]]; then
    base_model="${TOK}"
  fi
  if [[ -n "${base_model}" ]]; then
    python3 "${REPO_ROOT}/scripts/merge_adapter_shards.py" \
      --adapter-sharded-dir "${CKPT}_adapter_sharded" \
      --output-dir "${CKPT}_adapter" \
      --base-model "${base_model}"
  fi
fi

if [[ ! -f "${CKPT}/adapter_config.json" && -f "${CKPT}_adapter/adapter_config.json" ]]; then
  CKPT="${CKPT}_adapter"
fi
if [[ ! -f "${CKPT}/adapter_config.json" && -d "${CKPT}_adapter_sharded" ]]; then
  echo "[ERROR] Found sharded adapter checkpoint at ${CKPT}_adapter_sharded but no merged adapter." >&2
  echo "        Run scripts/merge_adapter_shards.py (torchrun) to produce ${CKPT}_adapter first." >&2
  exit 1
fi

MODEL_ARGS="pretrained=${CKPT},tokenizer=${TOK}"
if [[ -f "${CKPT}/adapter_config.json" ]]; then
  BASE=${TOK}
  if [[ -z "${BASE}" || "${BASE}" == "${CKPT}" || "${BASE}" == "${ORIG_CKPT}" ]]; then
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

HAS_LANGUAGE_LIST="false"
if [[ -f "${CKPT}/adapter_config.json" ]]; then
  HAS_LANGUAGE_LIST=$(python3 - <<'PY' "${CKPT}/adapter_config.json"
import json
import sys
cfg = json.load(open(sys.argv[1]))
print("true" if cfg.get("language_list") else "false")
PY
)
fi

USE_LANG="false"
if [[ "${USE_LANG_WRAPPER}" == "true" ]]; then
  USE_LANG="true"
elif [[ "${USE_LANG_WRAPPER}" == "auto" && "${HAS_LANGUAGE_LIST}" == "true" ]]; then
  USE_LANG="true"
fi

if [[ "${USE_LANG}" == "true" && -f "${CKPT}/adapter_config.json" ]]; then
  WRAPPER_ARGS=(
    "--checkpoint" "${CKPT}"
    "--tokenizer" "${TOK_USE}"
    "--tasks" "${TASKS}"
    "--batch-size" "${BS}"
    "--output-dir" "${OUTDIR}"
    "--mode" "${LANG_MODE}"
    "--wandb-args" "${WANDB_ARGS}"
  )
  if [[ -n "${LIMIT}" ]]; then
    WRAPPER_ARGS+=("--limit" "${LIMIT}")
  fi
  if [[ "${LOG_ROUTER_METRICS}" == "true" ]]; then
    WRAPPER_ARGS+=("--log-router-metrics")
  fi
  python3 "${REPO_ROOT}/scripts/lm_eval_language_ids.py" "${WRAPPER_ARGS[@]}" ${EXTRA}
else
  "${LM_EVAL_BIN}" \
    --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BS}" \
    --output_path "${OUTFILE}" \
    --wandb_args "${WANDB_ARGS}" \
    ${EXTRA}
fi
