#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/sample_data/generate_sampling_plans.py" \
  --tier-dir "$ROOT/tools/two_stage_clustering" \
  --fineweb-csv "$ROOT/data_prep/base_data/fineweb2-language-distribution.csv" \
  --output-dir "$ROOT/sample_data"
