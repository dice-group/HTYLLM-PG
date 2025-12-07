#!/usr/bin/env bash
set -euo pipefail

# inputs
DISTANCE_NPZ="data_prep/processed_artifacts/lang2vec_all_distances.npz"
METADATA_CSV="data_prep/base_data/fineweb2-language-distribution.csv"
MIN_DOCUMENTS=500

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
    --distance-npz "${DISTANCE_NPZ}" \
    --metadata-csv "${METADATA_CSV}" \
    --min-documents "${MIN_DOCUMENTS}" \
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
  python tools/farthest_points.py \
    --distance-npz "${DEMO_DIST}" \
    --coordinate-csv "${DEMO_COORDS}" \
    --k 4 \
    --target-total 72 \
    --output-image "${out_png}" \
    --output-umap-image "${out_umap}" \
    --json-output "${out_json}" \
    --show-all
fi

echo "All runs completed. Results stored in ${RESULT_DIR}/"
