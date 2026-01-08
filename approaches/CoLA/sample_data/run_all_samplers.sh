#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  # "tier1|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier1_12langs.csv|normal|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_12_tier_samples"
  # "tier1|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier1_12langs.csv|tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/tokenizer_cola_12_tier_samples"
  # "tier2|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier2_72langs.csv|normal|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_72_tier_samples"
  # "tier2|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier2_72langs.csv|tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/tokenizer_cola_72_tier_samples"
  # "tier3|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier3_200langs.csv|normal|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_200_tier_samples"
  "tier3|/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/sample_data/generate_sample_plan/sampling_plan_tier3_200langs.csv|tokenizer|/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/tokenizer_cola_200_tier_samples"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tier plan mode outdir <<< "${cfg}"
  flag=""
  [[ "${mode}" == "tokenizer" ]] && flag="--tokenizer-training"
  job="fw2_${tier}_${mode}"
  sbatch --export=ALL,OUTPUT_DIR="${outdir}" --job-name "${job}" "run_slurm_sampler.sh" "${plan}" "${flag}"
done
