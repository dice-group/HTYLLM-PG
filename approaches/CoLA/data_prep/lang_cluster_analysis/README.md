# Lang Cluster Analysis Toolkit

This directory bundles three light-weight scripts that complement the FLORES embedding work with typology-based insights:

1. `check_lang2vec_coverage.py` – reports which FLORES languages have Lang2Vec coverage (feature sets + distance matrices) so you know when typological features are available.
2. `cluster_lang2vec_distances.py` – clusters FLORES languages using Lang2Vec distance matrices (genetic, geographic, phonological, etc.).
3. `plot_lang2vec_clusters.py` – projects Lang2Vec distances to 2D (MDS), colors each point by language family, and prints the hierarchical clusters.

Combined with the FLORES embeddings produced by `embed_flores_langs.py` for both LLaMA‑3.1‑8B and Glot500, these scripts let you compare deep-model similarity with typological clustering.

## Dependencies
- Python packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.
- Local Lang2Vec code bundled under `data_prep/lang_cluster_analysis/lang2vec`.

To install Lang2Vec locally:
```bash
cd data_prep/lang_cluster_analysis/lang2vec
pip install -e .
```

## Script summary
- **check_lang2vec_coverage.py** – run with `python check_lang2vec_coverage.py`. Prints coverage stats and tips.
- **cluster_lang2vec_distances.py** – run with flags, e.g. `python cluster_lang2vec_distances.py --distance-type genetic --clusters 12`. Prints language memberships per cluster.
- **plot_lang2vec_clusters.py** – run with flags, e.g. `python plot_lang2vec_clusters.py --distance-type genetic --clusters 20 --output lang2vec_genetic.png`. Saves a 2D MDS scatter plot with language-family colors.

Use these outputs alongside the LLaMA‑3.1‑8B and Glot500 FLORES embeddings to run multi-view cluster analysis.
