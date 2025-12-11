# Farthest-Point Language Clustering

This helper builds concise language tiers from a precomputed distance matrix (lang2vec) by:

- selecting farthest seeds via greedy farthest-point sampling (maximizing pairwise distances),
- assigning every other language to the single seed that is strictly closer than any other seed (if a language is not uniquely closest, it stays unassigned and the script aborts), and
- producing summary plots (MDS + optional UMAP) plus JSON metadata describing the best `K` and every cluster’s allocations.

## Key invariants

- Seeds (`best_codes`) define the centers. Every neighbor listed under `best_neighbors[seed]` is closer to that seed than to any other seed (1e-6 tolerance), so clusters do not overlap.
- Tier sizes are enforced via `--target-total` (or legacy `--neighbors-per-language`); if the strict rule cannot supply that many languages, the run fails early so you can relax the target or drop filters.

## Running

```bash
python tools/farthest_points.py \
  --distance-npz data_prep/processed_artifacts/lang2vec_all_distances.npz \
  --k-min 4 --k-max 16 --target-total 72 \
  --metadata-csv data_prep/base_data/fineweb2-language-distribution.csv \
  --min-documents 500 \
  --output-image data_prep/processed_artifacts/farthest_points_k72.png \
  --output-umap-image data_prep/processed_artifacts/farthest_points_k72_umap.png \
  --json-output data_prep/processed_artifacts/farthest_points_k72.json \
  --show-all
```

Arguments:
- `--target-total / --neighbors-per-language`: controls the tier size; the script auto-distributes neighbors per seed.
- `--k` or `--k-min/--k-max`: fixed K or sweep range; the highest-scoring configuration becomes `best_k`.
- `--metadata-csv` + `--min-documents`: optional filters (e.g., include only languages with ≥500 documents).
- `--show-all`: plot all languages in grey with cluster members color-coded; otherwise only the cluster subset is drawn.

The JSON output contains:
- `best_k`, `best_codes`, and `best_neighbors` (cluster definition),
- `quality_metrics` (pairwise spread + neighbor coherence), and
- `per_k` entries for every evaluated K (useful for debugging or manual inspection).

## Tests

Run `pytest tests/test_farthest_points.py` to regress:
- farthest-point selection (size, rejection cases, quality vs random),
- neighbor allocation (unit tests for `assign_neighbors`),
- CLI smoke tests (synthetic matrices, coordinate CSV mode, limit flags),
- real-data integration checks (targets 72 and 200) validating that every neighbor is indeed closest to the assigned seed.

These tests ensure that any change preserves the strict “closest-seed” guarantee before regenerating tiers. ***
