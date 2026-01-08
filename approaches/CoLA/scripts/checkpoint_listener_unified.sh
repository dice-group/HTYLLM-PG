#!/bin/bash
#SBATCH --job-name=ckpt-listener
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=168:00:00
# Unified checkpoint listener: scans multiple roots and submits lm-eval jobs for new adapters.
# Options: --watch-root/--roots-file, --tasks/--limit, --marker-tag/--force.
# Logs: listener log in --log-root and per-run eval logs in subfolders.

set -euo pipefail

TASKS="belebele_eng_Latn"
BS="auto"
TOK=""
POLL=120
MIN_AGE=300
OUT_SUBDIR="lm_eval"
IGNORE_EXISTING=false
FORCE=false
LIMIT=""
LOG_ROOT="/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/scripts/listener_logs"
SCRIPT=""
ROOTS_FILE=""
WATCH_ROOTS=(
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/" # watch all
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
    --limit) LIMIT=$2; shift 2; continue;;
    --log-root) LOG_ROOT=$2; shift 2; continue;;
    --marker-tag) MARK_TAG=$2; shift 2; continue;;
    --ignore-existing) IGNORE_EXISTING=true; shift; continue;;
    --force) FORCE=true; shift; continue;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

find_eval_script() {
  local base=$1
  local cand=""
  while [[ -n "$base" && "$base" != "/" ]]; do
    cand="${base}/scripts/lm_eval_checkpoint_adapter_only.sh"
    [[ -f "$cand" ]] && { echo "$cand"; return 0; }
    cand="${base}/approaches/CoLA/scripts/lm_eval_checkpoint_adapter_only.sh"
    [[ -f "$cand" ]] && { echo "$cand"; return 0; }
    base=$(dirname "$base")
  done
  return 1
}

if [[ -z "$SCRIPT" || ! -f "$SCRIPT" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT="$(find_eval_script "$SLURM_SUBMIT_DIR")" || true
  fi
  if [[ -z "$SCRIPT" && -n "${REPO_ROOT:-}" ]]; then
    SCRIPT="$(find_eval_script "$REPO_ROOT")" || true
  fi
  if [[ -z "$SCRIPT" ]]; then
    SCRIPT="$(find_eval_script "$(pwd)")" || true
  fi
  if [[ -z "$SCRIPT" ]]; then
    echo "lm_eval_checkpoint_adapter_only.sh not found; pass --eval-script" >&2
    exit 1
  fi
fi

[[ ${#WATCH_ROOTS[@]} -eq 0 && -z "$ROOTS_FILE" ]] && { echo "Provide --watch-root or --roots-file"; exit 1; }

if [[ -z "$MARK_TAG" ]]; then
  MARK_TAG=$(echo -n "${TASKS}|${LIMIT}" | cksum | awk '{print $1}')
fi

mkdir -p "$LOG_ROOT"
listener_log="${LOG_ROOT}/listener.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  listener_log="${LOG_ROOT}/listener_${SLURM_JOB_ID}.log"
fi
exec > >(tee -a "$listener_log") 2>&1

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
  local run_label
  run_label=$(basename "$run_dir")
  local ckpt_label
  ckpt_label=$(basename "$target")
  local eval_out_dir="${run_dir}/${OUT_SUBDIR}/${ckpt_label}"
  local job_name="lm-eval_${ckpt_label}"
  local log_dir="${LOG_ROOT}/${run_label}"
  local job_log="${log_dir}/${job_name}_%j.log"
  local script_dir
  local repo_root=""
  script_dir=$(dirname "$SCRIPT")
  if [[ -f "${script_dir}/../scripts/lm_eval_language_ids.py" ]]; then
    repo_root=$(cd "${script_dir}/.." && pwd)
  fi
  mkdir -p "$eval_out_dir"
  mkdir -p "$log_dir"
  echo "[INFO] submit eval ckpt=${target} out=${eval_out_dir} tasks=${TASKS} limit=${LIMIT:-none} script=${SCRIPT}"
  local -a sbatch_cmd=(sbatch)
  if [[ -n "$LIMIT" ]]; then
    sbatch_cmd+=(--export=ALL,LM_EVAL_LIMIT="${LIMIT}")
  fi
  if [[ -n "$repo_root" ]]; then
    sbatch_cmd+=(--chdir="${repo_root}")
  fi
  sbatch_cmd+=(
    --job-name="${job_name}" \
    --output="${job_log}" \
    "$SCRIPT" \
    --checkpoint "$target" \
    --output-dir "$eval_out_dir" \
    --tasks "$TASKS" \
    --batch-size "$BS" \
    --wandb-prefix "$run_label" \
    --wandb-group "$run_label" \
    ${TOK:+--tokenizer "$TOK"} \
  )
  local job_out
  job_out=$("${sbatch_cmd[@]}")
  echo "[INFO] ${job_out} (ckpt=${target})"
  mark_seen "$target"
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
      if is_seen "$target"; then
        echo "[INFO] skip already evaluated: ckpt=${target} marker=${MARK_TAG}"
        continue
      fi
      is_old_enough "$ckpt" || continue
      submit_eval "$target"
    done
  done < <(load_roots)
  sleep "$POLL"
done
