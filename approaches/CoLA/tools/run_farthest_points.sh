#!/usr/bin/env bash
set -euo pipefail

# inputs
METADATA_CSV="data_prep/base_data/fineweb2-language-distribution.csv"
MIN_DOCUMENTS=500
LANG2VEC_DISTANCE="genetic"

RESULT_DIR="tools/farthest_runs"
mkdir -p "${RESULT_DIR}"

#drop current results to not confudse with new results
rm -f "${RESULT_DIR}/"*

run_fp() {
  local label=$1
  shift
  local out_png="${RESULT_DIR}/${label}.png"
  local out_umap="${RESULT_DIR}/${label}_umap.png"
  local out_json="${RESULT_DIR}/${label}.json"
  echo ">>> Running ${label}..."
  python tools/farthest_points.py \
    --metadata-csv "${METADATA_CSV}" \
    --min-documents "${MIN_DOCUMENTS}" \
    --lang2vec-distance "${LANG2VEC_DISTANCE}" \
    --output-image "${out_png}" \
    --output-umap-image "${out_umap}" \
    --json-output "${out_json}" \
    --show-all \
    "$@"
}

# 12-language tier (fixed K=4)
run_fp "tier12_k4" \
  --target-total 12 \
  --k 4

# 72-language tier (sweep K=4..12)
run_fp "tier72_k4-12" \
  --target-total 72 \
  --k-min 4 \
  --k-max 12

# 200-language tier (sweep K=4..16)
run_fp "tier200_k4-16" \
  --target-total 200 \
  --k-min 4 \
  --k-max 16

# Demo/test run using the bundled synthetic matrix, this helped me to verify the algo works as inteded
DEMO_DIST="data_prep/processed_artifacts/farthest_points_demo_distances.npz"
DEMO_COORDS="data_prep/processed_artifacts/farthest_points_demo_coords.csv"
if [[ -f "${DEMO_DIST}" ]]; then
  label="demo_test"
  out_png="${RESULT_DIR}/${label}.png"
  out_umap="${RESULT_DIR}/${label}_umap.png"
  out_json="${RESULT_DIR}/${label}.json"
  echo ">>> Running ${label} (demo)..."
  DEMO_ARGS=(
    --distance-npz "${DEMO_DIST}"
    --k 4
    --target-total 72
    --output-image "${out_png}"
    --output-umap-image "${out_umap}"
    --json-output "${out_json}"
    --show-all
  )
  if [[ -f "${DEMO_COORDS}" ]]; then
    DEMO_ARGS+=(--coordinate-csv "${DEMO_COORDS}")
  fi
  python tools/farthest_points.py "${DEMO_ARGS[@]}"
fi

echo "All runs completed. Results stored in ${RESULT_DIR}/"
