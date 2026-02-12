#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"


NO_IDS_DIRS=( #### TODO ####
  # # baselines 10% data: cola flat, hydra flat, lora baseline
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-flat_20260108_054502"
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/lora_lora-baseline_20260108_054502"

  # # # base lines full data
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/cola_colaflat_20260108_055323"
)

BOTH_DIRS=(
  # 1) 10 percent tier
  
  # cola
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-headbias_20260108_054502" # retry
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-hard_20260108_054502"     # failed
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-lpr_20260108_054502"      # this drastically underperformed and didnt recover from loss incidents # finsihed
  
  # hydralora
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-hard_20260108_054502"            # retry
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr_20260108_054502"             # failed
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr-expert-only_20260108_054502" # finished
  
  # # 2) Full data tier
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/cola_colaexp-lpr_20260108_055323"      # finished 
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/cola_colaexp-headbias_20260112_004550" # this was aborted at 55% to save compute for other subgroup # finished
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/hydra_hydra-exp-lpr_20260116_094457"   # this only ran for 14% # finished
)

# Defaults (override via env if needed)
EVAL_TIME="${EVAL_TIME:-72:00:00}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_GPU_TYPE="${EVAL_GPU_TYPE:-h100}"
EVAL_PARTITION="${EVAL_PARTITION:-gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-htyllm-adapter-lpr-200_lang_cola_eval}"
LOG_ROOT="${LOG_ROOT:-${PWD}/logs}"
LOG_ROOT_NO_IDS="${LOG_ROOT_NO_IDS:-${LOG_ROOT}/no_ids}"
LOG_ROOT_WITH_IDS="${LOG_ROOT_WITH_IDS:-${LOG_ROOT}/with_ids}"
TASKS_FILE="${TASKS_FILE:-${REPO_ROOT}/configs/lm_eval_tasks_200_langs_with_flores.txt}"
TASKS_SPLIT_DIR="${TASKS_SPLIT_DIR:-${LOG_ROOT}/tasks_split}"
BELEBELE_CHUNKS="${BELEBELE_CHUNKS:-1}"
FLORES_CHUNKS="${FLORES_CHUNKS:-4}"
TASKS_MANIFEST="${TASKS_MANIFEST:-${TASKS_SPLIT_DIR}/tasks_manifest.tsv}"
ENABLE_SPLIT="${ENABLE_SPLIT:-1}"

if [[ "${ENABLE_SPLIT}" == "1" ]]; then
  python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/split_tasks.py" \
    --tasks-file "${TASKS_FILE}" \
    --out-dir "${TASKS_SPLIT_DIR}" \
    --belebele-chunks "${BELEBELE_CHUNKS}" \
    --flores-chunks "${FLORES_CHUNKS}" \
    --manifest "${TASKS_MANIFEST}"
else
  mkdir -p "${TASKS_SPLIT_DIR}"
  printf "all\t%s\n" "${TASKS_FILE}" > "${TASKS_MANIFEST}"
fi

mapfile -t TASK_MANIFEST_LINES < <(cat "${TASKS_MANIFEST}")

for entry in "${TASK_MANIFEST_LINES[@]}"; do
  chunk_tag="${entry%%$'\t'*}"
  chunk_file="${entry#*$'\t'}"
  chunk_log_no_ids="${LOG_ROOT_NO_IDS}/${chunk_tag}"
  chunk_log_with_ids="${LOG_ROOT_WITH_IDS}/${chunk_tag}"
  chunk_output_subdir="lm_eval_latest/${chunk_tag}"

  if ((${#NO_IDS_DIRS[@]})); then
    LM_EVAL_LANG_MODE=no_ids \
    python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/submit_latest_eval.py" \
      --paths "${NO_IDS_DIRS[@]}" \
      --tasks-file "${chunk_file}" \
      --output-subdir "${chunk_output_subdir}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name-suffix "${chunk_tag}" \
      --log-root "${chunk_log_no_ids}" \
      --eval-partition "${EVAL_PARTITION}" \
      --eval-time "${EVAL_TIME}" \
      --eval-gpus "${EVAL_GPUS}" \
      --eval-gpu-type "${EVAL_GPU_TYPE}"
  fi

  if ((${#BOTH_DIRS[@]})); then
    LM_EVAL_LANG_MODE=no_ids \
    python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/submit_latest_eval.py" \
      --paths "${BOTH_DIRS[@]}" \
      --tasks-file "${chunk_file}" \
      --output-subdir "${chunk_output_subdir}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name-suffix "${chunk_tag}" \
      --log-root "${chunk_log_no_ids}" \
      --eval-partition "${EVAL_PARTITION}" \
      --eval-time "${EVAL_TIME}" \
      --eval-gpus "${EVAL_GPUS}" \
      --eval-gpu-type "${EVAL_GPU_TYPE}"

    LM_EVAL_LANG_MODE=with_ids \
    python3 "${REPO_ROOT}/scripts/eval/latest_checkpoint/submit_latest_eval.py" \
      --paths "${BOTH_DIRS[@]}" \
      --tasks-file "${chunk_file}" \
      --output-subdir "${chunk_output_subdir}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name-suffix "${chunk_tag}" \
      --log-root "${chunk_log_with_ids}" \
      --eval-partition "${EVAL_PARTITION}" \
      --eval-time "${EVAL_TIME}" \
      --eval-gpus "${EVAL_GPUS}" \
      --eval-gpu-type "${EVAL_GPU_TYPE}"
  fi
done
