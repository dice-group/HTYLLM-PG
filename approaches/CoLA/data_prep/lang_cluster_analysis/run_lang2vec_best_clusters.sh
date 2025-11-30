
# this runs Lang2Vec clustering over all FLORES languages, 
# lets auto-k pick the best number of clusters, 
# writes plots plus the top 4 clusters × 3 languages with the highest silhouette scores

python data_prep/lang_cluster_analysis/cluster_lang2vec_distances.py \
  --distance-type genetic \
  --auto-k \
  --min-k 4 \
  --max-k 12 \
  --output data_prep/processed_artifacts/clusters_lang2vec_genetic_auto.json \
  --plots-dir data_prep/processed_artifacts/plots/lang2vec_genetic_auto \
  --top-per-cluster 3 \
  --best-clusters 4 \
  --top-per-cluster-output data_prep/processed_artifacts/clusters_lang2vec_genetic_auto_best4x3.json \
  --top-n 24 \
  --top-n-output data_prep/processed_artifacts/clusters_lang2vec_genetic_auto_top24.json
