# Language Selection Requirements

- Budgets: define four ascending subsets (≈12, 72–95, 200, 635 langs) aligned with the training tiers in `docs/decide_token_budget.md`, each subset being a superset of the smaller tier.
- Families: for every tier choose `X` families that are maximally distant in Lang2Vec space (via medoids from the chosen distance type) while preserving previously selected families.
- Per family: select `Y` tightly clustered languages (low intra-family distance) so that `X × Y` matches the tier’s size target; when possible balance high-/mid-/low-resource members (for expert targeting), skipping families lacking enough candidates and falling back to the next-best family.
- Scoring: evaluate candidate `(X, Y)` combos with `score = α·inter – β·intra`, where inter is the average distance across the selected languages between families and intra the mean within-family distance; list all viable combos ordered by score per tier.
- Signals: base distances on Lang2Vec’s precomputed matrices (`genetic`, `syntactic`, `phonological`, `inventory`, etc.). Raw WALS/SSWL features are largely missing, but `fam`, `geo`, and the KNN feature sets cover all 1,804 overlapping languages.

### Feature Choice Rationale

- Use the Lang2Vec precomputed distances (genetic, syntactic, phonological, inventory) as the primary separation signal; each is backed by complete KNN-derived feature vectors for every overlapping language => so we have all information
- Prefer the `syntax_knn`, `phonology_knn`, and `inventory_knn` feature sets when inspecting or debugging specific languages, since they are the only raw vectors with full coverage (alongside `fam`, `geo`, and `id`).
- Avoid raw WALS/SSWL/Ethnologue feature sets (mostly empty) and PHOIBLE subsets (cover <15% of languages), unless the analysis is restricted to their limited coverage.
- I assume Distance -> feature mapping is like this, but i cannot prove since they are precomputed.
  - `genetic`: `fam` + `id` (family membership / phylogeny one-hot vectors).
  - `geographic`: `geo` (latitude/longitude coordinates).
  - `syntactic`: union of `syntax_wals`, `syntax_sswl`, `syntax_ethnologue`, `syntax_knn`, `syntax_average`.
  - `phonological`: union of `phonology_wals`, `phonology_ethnologue`, `phonology_knn`, `phonology_average`.
  - `inventory`: union of `inventory_ethnologue`, all `inventory_phoible_*` sets, `inventory_knn`, `inventory_average`.
  - `featural`: concatenation of the full syntactic + phonological + inventory feature vectors.
- One can check the implementation https://github.com/antonisa/lang2vec/blob/master/lang2vec/lang2vec.py
- the `.npz` distance files in `lang2vec/data/` were likely generated such definitions 
### Notes on KNN Feature Sets

- The original URIEL/PHOIBLE/WALS feature inventories are sparse for our languages; the KNN variants fill those gaps by predicting missing typological values.
- Per the Lang2Vec paper ([Littell et al., 2017](https://aclanthology.org/E17-2002.pdf)), missing values are predicted via weighted 10-nearest-neighbor classification, where neighbors are found using the average of genetic, geographic, and featural distances between languages. This approach reaches roughly 93% accuracy (10-fold cross-validation) and yields dense `syntax_knn`, `phonology_knn`, and `inventory_knn` vectors for every language in our overlap, making the downstream distance matrices reliable.

## Required Inputs

- FineWeb per-language metadata (code, family, resource size) from `data_prep/base_data/fineweb2-language-distribution.csv`.
- Lang2Vec distance matrices (at least one of genetic/syntactic/phonological/inventory/featural) plus medoid extraction per family.
- Resource-level stats (e.g., document counts) to tag languages as high/mid/low for expert-aware per-family balancing.
