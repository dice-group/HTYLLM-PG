#!/bin/bash
set -euo pipefail

# Sampler for each lang tier: Submit tier-specific sampling jobs with tailored hardware settings.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAN_DIR="${REPO_ROOT}/data_prep/processed_artifacts"

declare -A TIER_CONFIGS=(
  # tier_id="plan_filename|array_max|cpus|mem|time"
  [tier1]="sampling_plan_tier1_12langs.csv|11|2|8G|04:00:00"
  [tier2]="sampling_plan_tier2_72langs.csv|71|4|24G|12:00:00"
  [tier3]="sampling_plan_tier3_200langs.csv|199|4|32G|24:00:00"
)

submit_job() {
  local tier="$1"
  local plan="$2"
  local array_max="$3"
  local cpus="$4"
  local mem="$5"
  local wall="$6"

  if [[ ! -f "${plan}" ]]; then
    echo "Plan ${plan} missing. Run sample_data/generate_sampling_plans.py first." >&2
    exit 1
  fi

  echo "Submitting ${tier} sampling job..."
  sbatch \
    --job-name="fw2_${tier}" \
    --array="0-${array_max}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${wall}" \
    run_slurm_sampler.sh "${plan}"
}

for tier in tier1 tier2 tier3; do
  IFS="|" read -r filename array_max cpus mem wall <<<"${TIER_CONFIGS[$tier]}"
  plan_path="${PLAN_DIR}/${filename}"
  submit_job "${tier}" "${plan_path}" "${array_max}" "${cpus}" "${mem}" "${wall}"
done
