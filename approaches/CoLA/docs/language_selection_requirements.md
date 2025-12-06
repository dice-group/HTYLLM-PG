# Language Selection Requirements

- Budgets: define four ascending subsets (≈12, 72–95, 200, 635 langs) aligned with the training tiers in `docs/decide_token_budget.md`, each subset being a superset of the smaller tier.
- Families: for every tier choose `X` families that are maximally distant in Lang2Vec space (via medoids from the chosen distance type) while preserving previously selected families.
- Per family: select `Y` tightly clustered languages (low intra-family distance) so that `X × Y` matches the tier’s size target; when possible balance high-/mid-/low-resource members (for expert targeting), skipping families lacking enough candidates and falling back to the next-best family.
- Scoring: evaluate candidate `(X, Y)` combos with `score = α·inter – β·intra`, where inter is the average distance across the selected languages between families and intra the mean within-family distance; list all viable combos ordered by score per tier.
- Signals: base distances on Lang2Vec’s precomputed matrices (`genetic`, `syntactic`, `phonological`, `inventory`, etc.). Raw WALS/SSWL features are largely missing, but `fam`, `geo`, and the KNN feature sets cover all 1,804 overlapping languages.

## Required Inputs

- FineWeb per-language metadata (code, family, resource size) from `data_prep/base_data/fineweb2-language-distribution.csv`.
- Lang2Vec distance matrices (at least one of genetic/syntactic/phonological/inventory/featural) plus medoid extraction per family.
- Resource-level stats (e.g., document counts) to tag languages as high/mid/low for expert-aware per-family balancing.
