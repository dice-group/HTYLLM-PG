# Lang Cluster Analysis Toolkit

Goal: pick language clusters for mixture-of-experts fine-tuning of LLaMA‑3.1‑8B. We triangulate three signals:
1. **Model behavior** – generate FLORES embeddings with `embed_flores_langs.py --model-key llama31_8b` to see how the base LLaMA groups languages.
2. **Coverage-aware baseline** – run the same script with `--model-key glot500` to capture languages LLaMA underrepresents.
3. **Typology prior** – compare both embedding spaces against Lang2Vec’s linguistic distances using the scripts here. Agreements strengthen our expert groupings; disagreements highlight where extra supervision may help.

This folder holds the Lang2Vec analysis scripts:

1. `check_lang2vec_coverage.py` – reports which FLORES languages have Lang2Vec coverage (feature sets + distance matrices) so you know when typological features are available.
2. `cluster_lang2vec_distances.py` – clusters FLORES languages using Lang2Vec distance matrices (genetic, geographic, phonological, etc.).
3. `plot_lang2vec_clusters.py` – projects Lang2Vec distances to 2D (MDS), colors each point by language family, and prints the hierarchical clusters.

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
- **check_lang2vec_coverage.py** – run with `python check_lang2vec_coverage.py`. Prints coverage stats and tips.
- **cluster_lang2vec_distances.py** – run with flags, e.g. `python cluster_lang2vec_distances.py --distance-type genetic --clusters 12`. Prints language memberships per cluster.
- **plot_lang2vec_clusters.py** – run with flags, e.g. `python plot_lang2vec_clusters.py --distance-type genetic --clusters 20 --output lang2vec_genetic.png`. Saves a 2D MDS scatter plot with language-family colors.

Use these outputs alongside the LLaMA‑3.1‑8B and Glot500 FLORES embeddings to run multi-view cluster analysis.
