#!/bin/bash
#SBATCH --job-name=ckpt-listener
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=168:00:00
#SBATCH --output=listener_logs/listener_%j.log

# Unified checkpoint listener: scans multiple roots and submits lm-eval jobs for new adapters.
# Options: --watch-root/--roots-file, --tasks/--limit, --marker-tag/--force.
# Logs: listener log in --log-root and per-run eval logs in subfolders.
# example usage: forces re-eval and limits to 100 sbatch ./checkpoint_listener_unified.sh --force --limit 100

set -euo pipefail

TASKS="belebele_eng_Latn"
BS="auto"
TOK=""
POLL=120
MIN_AGE=300
OUT_SUBDIR="lm_eval"
IGNORE_EXISTING=false
FORCE=false
FORCE_ONCE=false
LIMIT=""
LOG_ROOT="/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/scripts/listener_logs"
SCRIPT=""
SYNC_INTERVAL=120
SYNC_PROJECT="htyllm-adapter-lpr-200_lang_cola_eval"
SYNC_SUFFIX=""
SUMMARY_PROJECT_SUFFIX="${LM_EVAL_SUMMARY_SUFFIX:-_summary}"
SYNC_SCRIPT=""
SYNC_PYTHON="python3"
ROOTS_FILE=""
WATCH_ROOTS=(
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/" # watch all
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-hard_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-headbias_20260108_054502"
   "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-lpr_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaflat_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-hard_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr-expert-only_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-flat_20260108_054502"
  # "/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/lora_lora-baseline_20260108_054502"
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
    --sync-interval) SYNC_INTERVAL=$2; shift 2; continue;;
    --sync-project) SYNC_PROJECT=$2; shift 2; continue;;
    --sync-suffix) SYNC_SUFFIX=$2; shift 2; continue;;
    --sync-script) SYNC_SCRIPT=$2; shift 2; continue;;
    --sync-python) SYNC_PYTHON=$2; shift 2; continue;;
    --marker-tag) MARK_TAG=$2; shift 2; continue;;
    --ignore-existing) IGNORE_EXISTING=true; shift; continue;;
    --force) FORCE=true; FORCE_ONCE=true; shift; continue;;
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

find_sync_script() {
  local base=$1
  local cand=""
  while [[ -n "$base" && "$base" != "/" ]]; do
    cand="${base}/scripts/wandb_summary_sync.py"
    [[ -f "$cand" ]] && { echo "$cand"; return 0; }
    cand="${base}/approaches/CoLA/scripts/wandb_summary_sync.py"
    [[ -f "$cand" ]] && { echo "$cand"; return 0; }
    base=$(dirname "$base")
  done
  return 1
}

if [[ -z "$SYNC_SCRIPT" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SYNC_SCRIPT="$(find_sync_script "$SLURM_SUBMIT_DIR")" || true
  fi
  if [[ -z "$SYNC_SCRIPT" && -n "${REPO_ROOT:-}" ]]; then
    SYNC_SCRIPT="$(find_sync_script "$REPO_ROOT")" || true
  fi
  if [[ -z "$SYNC_SCRIPT" ]]; then
    SYNC_SCRIPT="$(find_sync_script "$(pwd)")" || true
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
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
exec > >(tee -a "$listener_log") 2>&1
fi

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

sync_summaries() {
  [[ "$SYNC_INTERVAL" -le 0 ]] && return 0
  [[ -z "$SYNC_SCRIPT" || ! -f "$SYNC_SCRIPT" ]] && { echo "[WARN] sync script not found; skipping"; return 0; }
  declare -A seen_runs=()
  while IFS= read -r root; do
    [[ -z "$root" ]] && continue
    mapfile -t RUN_DIRS < <(find "$root" -maxdepth 2 -type d -name "$OUT_SUBDIR" | sort -V)
    for run_dir in "${RUN_DIRS[@]}"; do
      [[ -z "$run_dir" ]] && continue
      if [[ -n "${seen_runs[$run_dir]+x}" ]]; then
        continue
      fi
      seen_runs["$run_dir"]=1
      local run_label
      run_label=$(basename "$(dirname "$run_dir")")
      local args="project=${SYNC_PROJECT},name=${run_label}"
      echo "[INFO] summary sync run_dir=${run_dir} wandb_args=${args}"
      if [[ -n "$SYNC_SUFFIX" ]]; then
        WANDB_SUMMARY_SUFFIX="${SUMMARY_PROJECT_SUFFIX}" "$SYNC_PYTHON" "$SYNC_SCRIPT" --run-dir "$run_dir" --wandb-args "$args" --suffix "$SYNC_SUFFIX" || true
      else
        WANDB_SUMMARY_SUFFIX="${SUMMARY_PROJECT_SUFFIX}" "$SYNC_PYTHON" "$SYNC_SCRIPT" --run-dir "$run_dir" --wandb-args "$args" || true
      fi
    done
  done < <(load_roots)
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
  local eval_dir
  eval_dir="$(dirname "$target")/${OUT_SUBDIR}/$(basename "$target")"
  mkdir -p "$eval_dir"
  touch "${target}/.eval_submitted_${MARK_TAG}" || true
  touch "${eval_dir}/.eval_submitted_${MARK_TAG}"
}

is_seen() {
  local target=$1
  if [[ "$FORCE" == "true" ]]; then
    return 1
  fi
  local eval_dir
  eval_dir="$(dirname "$target")/${OUT_SUBDIR}/$(basename "$target")"
  local marker_present=false
  if [[ -f "${target}/.eval_submitted_${MARK_TAG}" \
      || -f "${target}/.eval_done_${MARK_TAG}" \
      || -f "${target}/.eval_failed_${MARK_TAG}" \
      || -f "${eval_dir}/.eval_submitted_${MARK_TAG}" \
      || -f "${eval_dir}/.eval_done_${MARK_TAG}" \
      || -f "${eval_dir}/.eval_failed_${MARK_TAG}" ]]; then
    marker_present=true
  fi
  if [[ "$marker_present" == "true" && -d "${eval_dir}/.eval_lock_${MARK_TAG}" ]]; then
    rmdir "${eval_dir}/.eval_lock_${MARK_TAG}" 2>/dev/null || true
  fi
  [[ "$marker_present" == "true" || -d "${eval_dir}/.eval_lock_${MARK_TAG}" ]]
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
  local lock_dir="${eval_out_dir}/.eval_lock_${MARK_TAG}"
  if [[ "$FORCE" == "true" && -d "$lock_dir" ]]; then
    rmdir "$lock_dir" 2>/dev/null || true
  fi
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "[INFO] skip locked: ckpt=${target} marker=${MARK_TAG}"
    return
  fi
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
  local job_out=""
  if job_out=$("${sbatch_cmd[@]}"); then
    echo "[INFO] ${job_out} (ckpt=${target})"
    mark_seen "$target"
    rmdir "$lock_dir" 2>/dev/null || true
  else
    rmdir "$lock_dir" 2>/dev/null || true
    echo "[WARN] sbatch failed for ckpt=${target}" >&2
    return
  fi
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
  declare -A seen_targets=()
  while IFS= read -r root; do
    [[ -z "$root" ]] && continue
    mapfile -t CKPTS < <(
      find "$root" -type f -path '*/checkpoint-*_adapter/adapter_config.json' -printf '%h\n' | sort -V
    )
    for ckpt in "${CKPTS[@]}"; do
      target=$(resolve_adapter "$ckpt")
      [[ -z "$target" ]] && continue
      if [[ -n "${seen_targets[$target]+x}" ]]; then
        continue
      fi
      seen_targets["$target"]=1
      if is_seen "$target"; then
        echo "[INFO] skip already evaluated: ckpt=${target} marker=${MARK_TAG}"
        continue
      fi
      is_old_enough "$ckpt" || continue
      submit_eval "$target"
    done
  done < <(load_roots)
  if [[ "$SYNC_INTERVAL" -gt 0 ]]; then
    now=$(date +%s)
    if [[ -z "${LAST_SYNC_TS:-}" ]]; then
      LAST_SYNC_TS=0
    fi
    if (( now - LAST_SYNC_TS >= SYNC_INTERVAL )); then
      sync_summaries
      LAST_SYNC_TS=$now
    fi
  fi
  if [[ "$FORCE_ONCE" == "true" && "$FORCE" == "true" ]]; then
    echo "[INFO] force mode applied once; disabling for subsequent polls"
    FORCE=false
    FORCE_ONCE=false
  fi
  sleep "$POLL"
done
