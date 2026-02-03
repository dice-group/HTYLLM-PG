
# this runs Lang2Vec clustering over all FLORES languages, 
# lets auto-k pick the best number of clusters, 
# writes plots plus the top 4 clusters × 3 languages with the highest silhouette scores

OUTPUT_DIR="data_prep/processed_artifacts"
PLOTS_DIR="${OUTPUT_DIR}/plots/lang2vec_genetic_auto"
BASE_OUTPUT="${OUTPUT_DIR}/clusters_lang2vec_genetic_auto"
TOP_PER_CLUSTER=${TOP_PER_CLUSTER:-6}
BEST_CLUSTERS=${BEST_CLUSTERS:-4}
LOW_PERF_PER_CLUSTER=${LOW_PERF_PER_CLUSTER:-3}
LOW_PERF_THRESHOLD=${LOW_PERF_THRESHOLD:-0.35}
PERF_CSV=${PERF_CSV:-result_analysis/lm_eval/belebele_llama31_8b_accuracy.csv}
PRESET_PATH="data_prep/lang_cluster_analysis/presets/lang2vec_auto_best4x3.json"

python data_prep/lang_cluster_analysis/cluster_lang2vec_distances.py \
  --distance-type genetic \
  --auto-k \
  --min-k 4 \
  --max-k 12 \
  --output "${BASE_OUTPUT}.json" \
  --plots-dir "${PLOTS_DIR}" \
  --top-per-cluster "${TOP_PER_CLUSTER}" \
  --best-clusters "${BEST_CLUSTERS}" \
  --top-per-cluster-output "${BASE_OUTPUT}_best4x${TOP_PER_CLUSTER}.json" \
  --top-n 24 \
  --top-n-output "${BASE_OUTPUT}_top24.json" \
  --low-perf-csv "${PERF_CSV}" \
  --low-perf-threshold "${LOW_PERF_THRESHOLD}" \
  --low-perf-per-cluster "${LOW_PERF_PER_CLUSTER}" \
  --low-perf-output "${BASE_OUTPUT}_best4x3_lowacc.json" \
  --low-perf-report "${BASE_OUTPUT}_best4x3_lowacc_report.json" \
  --low-perf-plot "${PLOTS_DIR}/clusters_lang2vec_genetic_auto_selection.png"

if [[ -f "${BASE_OUTPUT}_best4x3_lowacc.json" ]]; then
  cp "${BASE_OUTPUT}_best4x3_lowacc.json" "${PRESET_PATH}"
  echo "Updated preset ${PRESET_PATH}"
fi
