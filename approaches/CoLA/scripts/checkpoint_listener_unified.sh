#!/bin/bash
#SBATCH --job-name=ckpt-listener
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=168:00:00
#SBATCH --output=logs/checkpoint_listener_%j.log

# this is one unified checkpoint listerner taht can watch multiple paths for checkpoint, rather than having one per path as before -> simplicity
# Watch one or more roots for new adapter checkpoints and submit lm-eval jobs.
# Use --roots-file for dynamic roots, and --ignore-existing to avoid backfill.
# Use --marker-tag or --force to control re-evals across different task sets.

set -euo pipefail

TASKS="belebele_eng_Latn"
BS="auto"
TOK=""
POLL=120
MIN_AGE=300
OUT_SUBDIR="lm_eval"
IGNORE_EXISTING=false
FORCE=false
SCRIPT="$(cd "$(dirname "$0")" && pwd)/lm_eval_checkpoint_adapter_only.sh"
ROOTS_FILE=""
WATCH_ROOTS=(
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/" # uncomment to watch all
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-hard_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-headbias_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-lpr_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-hard_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr-expert-only_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-flat_20260108_054502"
  "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/lora_lora-baseline_20260108_054502"
)
MARK_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-root) WATCH_ROOTS+=("$2"); shift 2; continue;;
    --roots-file) ROOTS_FILE=$2; shift 2; continue;;
    --eval-script) SCRIPT=$2; shift 2; continue;;
    --tasks) TASKS=$2; shift 2; continue;;
    --tokenizer) TOK=$2; shift 2; continue;;
    --batch-size) BS=$2; shift 2; continue;;
    --output-subdir) OUT_SUBDIR=$2; shift 2; continue;;
    --poll-interval) POLL=$2; shift 2; continue;;
    --min-age) MIN_AGE=$2; shift 2; continue;;
    --marker-tag) MARK_TAG=$2; shift 2; continue;;
    --ignore-existing) IGNORE_EXISTING=true; shift; continue;;
    --force) FORCE=true; shift; continue;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

[[ ${#WATCH_ROOTS[@]} -eq 0 && -z "$ROOTS_FILE" ]] && { echo "Provide --watch-root or --roots-file"; exit 1; }

if [[ -z "$MARK_TAG" ]]; then
  MARK_TAG=$(echo -n "$TASKS" | cksum | awk '{print $1}')
fi

mkdir -p logs

load_roots() {
  local -a roots=()
  for r in "${WATCH_ROOTS[@]}"; do
    [[ -n "$r" ]] && roots+=("$r")
  done
  if [[ -n "$ROOTS_FILE" && -f "$ROOTS_FILE" ]]; then
    while IFS= read -r line; do
      line="${line%%#*}"
      line="$(echo "$line" | xargs)"
      [[ -n "$line" ]] && roots+=("$line")
    done < "$ROOTS_FILE"
  fi
  printf "%s\n" "${roots[@]}"
}

is_old_enough() {
  local path=$1
  [[ "$MIN_AGE" -le 0 ]] && return 0
  local now mtime age
  now=$(date +%s)
  mtime=$(stat -c %Y "$path" 2>/dev/null || echo 0)
  age=$((now - mtime))
  [[ "$age" -ge "$MIN_AGE" ]]
}

resolve_adapter() {
  local ckpt=$1
  if [[ -f "${ckpt}/adapter_model.safetensors" || -f "${ckpt}/adapter_model.bin" ]]; then
    echo "$ckpt"
    return
  fi
  if [[ -f "${ckpt}_adapter/adapter_model.safetensors" || -f "${ckpt}_adapter/adapter_model.bin" ]]; then
    echo "${ckpt}_adapter"
    return
  fi
  echo ""
}

mark_seen() {
  local target=$1
  touch "${target}/.eval_submitted_${MARK_TAG}"
}

is_seen() {
  local target=$1
  if [[ "$FORCE" == "true" ]]; then
    return 1
  fi
  [[ -f "${target}/.eval_submitted_${MARK_TAG}" || -f "${target}/.eval_done_${MARK_TAG}" || -f "${target}/.eval_failed_${MARK_TAG}" ]]
}

submit_eval() {
  local target=$1
  local run_dir
  run_dir=$(dirname "$target")
  local ckpt_label
  ckpt_label=$(basename "$target")
  local eval_out_dir="${run_dir}/${OUT_SUBDIR}/${ckpt_label}"
  local job_name="lm-eval_${ckpt_label}"
  local job_log="${eval_out_dir}/${job_name}_%j.log"
  mkdir -p "$eval_out_dir"
  sbatch \
    --job-name="${job_name}" \
    --output="${job_log}" \
    "$SCRIPT" \
    --checkpoint "$target" \
    --output-dir "$eval_out_dir" \
    --tasks "$TASKS" \
    --batch-size "$BS" \
    ${TOK:+--tokenizer "$TOK"} \
  && mark_seen "$target"
}

if [[ "$IGNORE_EXISTING" == "true" ]]; then
  while IFS= read -r root; do
    [[ -z "$root" ]] && continue
    mapfile -t CKPTS < <(find "$root" -type d -name 'checkpoint-*' | sort -V)
    for ckpt in "${CKPTS[@]}"; do
      target=$(resolve_adapter "$ckpt")
      [[ -n "$target" ]] && mark_seen "$target"
    done
  done < <(load_roots)
fi

echo "[INFO] Unified listener started."

while true; do
  while IFS= read -r root; do
    [[ -z "$root" ]] && continue
    mapfile -t CKPTS < <(find "$root" -type d -name 'checkpoint-*' | sort -V)
    for ckpt in "${CKPTS[@]}"; do
      target=$(resolve_adapter "$ckpt")
      [[ -z "$target" ]] && continue
      is_seen "$target" && continue
      is_old_enough "$ckpt" || continue
      submit_eval "$target"
    done
  done < <(load_roots)
  sleep "$POLL"
done
