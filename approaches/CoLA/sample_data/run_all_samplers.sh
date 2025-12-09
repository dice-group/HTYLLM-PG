#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "tier1|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier1_12langs.csv|normal"
  "tier1|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier1_12langs.csv|tokenizer"
  "tier2|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier2_72langs.csv|normal"
  "tier2|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier2_72langs.csv|tokenizer"
  "tier3|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier3_200langs.csv|normal"
  "tier3|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier3_200langs.csv|tokenizer"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tier plan mode <<< "${cfg}"
  flag=""
  [[ "${mode}" == "tokenizer" ]] && flag="--tokenizer-training"
  job="fw2_${tier}_${mode}"
  sbatch --job-name "${job}" "run_slurm_sampler.sh" "${plan}" "${flag}"
done
