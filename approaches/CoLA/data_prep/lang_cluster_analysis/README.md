# Lang Cluster Analysis Toolkit

Goal: pick language clusters for mixture-of-experts fine-tuning of LLaMA‑3.1‑8B. We triangulate three signals:
1. **Model behavior** – generate FLORES embeddings with `embed_flores_langs.py --model-key llama31_8b` to see how the base LLaMA groups languages.
2. **Coverage-aware baseline** – run the same script with `--model-key glot500` to capture languages LLaMA underrepresents.
3. **Typology prior** – compare both embedding spaces against Lang2Vec’s linguistic distances using the scripts here. Agreements strengthen our expert groupings; disagreements highlight where extra supervision may help.

This folder holds the clustering/analysis scripts:

1. `cluster_embeddings.py` – clusters per-language FLORES embeddings (LLaMA or Glot500) with auto-k silhouette logic.
2. `check_lang2vec_coverage.py` – reports which FLORES languages have Lang2Vec coverage (feature sets + distance matrices) so you know when typological features are available.
3. `cluster_lang2vec_distances.py` – clusters FLORES languages using Lang2Vec distance matrices (genetic, geographic, phonological, etc.).
4. `plot_lang2vec_clusters.py` – projects Lang2Vec distances to 2D (MDS), colors each point by language family, and prints the hierarchical clusters.
5. `run_clustering_variants.py` – orchestrates the full pipeline, producing `clusters_llama31_8b.json`, `clusters_glot500.json`, and `clusters_lang2vec.json` under `processed_artifacts/`.

Workflow:
1. Compute LLaMA and Glot500 FLORES embeddings (per-language averages).
2. Run `check_lang2vec_coverage.py` to confirm typology support and distance coverage.
3. Use `cluster_lang2vec_distances.py` / `plot_lang2vec_clusters.py` to inspect typological neighborhoods.
4. Choose expert assignments where all three signals agree; investigate mismatches (Glot500/Lang2Vec similarities missing in LLaMA) as candidates for joint fine-tuning experts.
5. Fine-tune LLaMA experts with the selected language batches.

## Dependencies
- Python packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.
- Local Lang2Vec code bundled under `data_prep/lang_cluster_analysis/lang2vec`.

To install Lang2Vec locally:
```bash
cd data_prep/lang_cluster_analysis/lang2vec
pip install -e .
```

## Script summary
- **cluster_embeddings.py** – `python cluster_embeddings.py --input flores_embeddings_llama31_8b.csv --output clusters.json --auto-k --scale --plots-dir plots/llama31 --top-n 40`. Produces per-language cluster IDs from embedding CSVs plus silhouette/2D plots and, if desired, a filtered top-N subset (`--top-n-output` optional).
- **check_lang2vec_coverage.py** – `python check_lang2vec_coverage.py`. Prints coverage stats and tips.
- **cluster_lang2vec_distances.py** – `python cluster_lang2vec_distances.py --distance-type genetic --output clusters_lang2vec.json --auto-k --plots-dir plots/lang2vec --top-n 40`. Saves typology-driven cluster IDs with silhouette/MDS visualizations and optional top-N filtered subset for dense clusters. Use `--top-per-cluster N` plus `--best-clusters K` to grab a balanced subset of the strongest clusters, and optionally add `--low-perf-csv result_analysis/lm_eval/belebele_llama31_8b_accuracy.csv --low-perf-threshold 0.35 --low-perf-per-cluster 3 --low-perf-output clusters_lowacc.json --low-perf-plot plots/lang2vec/selection.png` to bias the final selection toward languages where the base model underperforms (e.g., ≤35 % Belebele accuracy) and generate a highlight visualization.
- **plot_lang2vec_clusters.py** – `python plot_lang2vec_clusters.py --distance-type genetic --clusters 20 --output lang2vec_genetic.png`. Saves a 2D MDS scatter plot with language-family colors.
- **run_clustering_variants.py** – `python run_clustering_variants.py` to regenerate all cluster artifacts in one go (skips embedding recompute if files already exist).

Use these outputs alongside the LLaMA‑3.1‑8B and Glot500 FLORES embeddings to run multi-view cluster analysis.
