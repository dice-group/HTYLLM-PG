#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"


NO_IDS_DIRS=(
  # baselines 10% data: cola flat, hydra flat, lora baseline
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-flat_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/lora_lora-baseline_20260108_054502"

  # # base lines full data
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/cola_colaflat_20260108_055323"
)

BOTH_DIRS=(
  # 1) 10 percent tier
  
  # cola
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-headbias_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-hard_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-lpr_20260108_054502" # this drastically underperformed and didnt recover from loss incidents
  
  # hydralora
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-hard_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr-expert-only_20260108_054502"
  
  # # 2) Full data tier
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/cola_colaexp-lpr_20260108_055323"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/cola_colaexp-headbias_20260112_004550" # this was aborted at 55% to save compute for other subgroup
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/hydra_hydra-exp-lpr_20260116_094457"   # this only ran for 14%
)

# Defaults (override via env if needed)
EVAL_TIME="${EVAL_TIME:-12:00:00}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_GPU_TYPE="${EVAL_GPU_TYPE:-h100}"
EVAL_PARTITION="${EVAL_PARTITION:-gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-htyllm-adapter-lpr-200_lang_cola_eval}"
LOG_ROOT="${LOG_ROOT:-${PWD}/logs}"
LOG_ROOT_NO_IDS="${LOG_ROOT_NO_IDS:-${LOG_ROOT}/no_ids}"
LOG_ROOT_WITH_IDS="${LOG_ROOT_WITH_IDS:-${LOG_ROOT}/with_ids}"

if ((${#NO_IDS_DIRS[@]})); then
  LM_EVAL_LANG_MODE=no_ids \
  python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/submit_latest_eval.py" \
    --paths "${NO_IDS_DIRS[@]}" \
    --wandb-project "${WANDB_PROJECT}" \
    --log-root "${LOG_ROOT_NO_IDS}" \
    --eval-partition "${EVAL_PARTITION}" \
    --eval-time "${EVAL_TIME}" \
    --eval-gpus "${EVAL_GPUS}" \
    --eval-gpu-type "${EVAL_GPU_TYPE}"
fi

if ((${#BOTH_DIRS[@]})); then
  LM_EVAL_LANG_MODE=no_ids \
  python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/submit_latest_eval.py" \
    --paths "${BOTH_DIRS[@]}" \
    --wandb-project "${WANDB_PROJECT}" \
    --log-root "${LOG_ROOT_NO_IDS}" \
    --eval-partition "${EVAL_PARTITION}" \
    --eval-time "${EVAL_TIME}" \
    --eval-gpus "${EVAL_GPUS}" \
    --eval-gpu-type "${EVAL_GPU_TYPE}"

  LM_EVAL_LANG_MODE=with_ids \
  python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/submit_latest_eval.py" \
    --paths "${BOTH_DIRS[@]}" \
    --wandb-project "${WANDB_PROJECT}" \
    --log-root "${LOG_ROOT_WITH_IDS}" \
    --eval-partition "${EVAL_PARTITION}" \
    --eval-time "${EVAL_TIME}" \
    --eval-gpus "${EVAL_GPUS}" \
    --eval-gpu-type "${EVAL_GPU_TYPE}"
fi
