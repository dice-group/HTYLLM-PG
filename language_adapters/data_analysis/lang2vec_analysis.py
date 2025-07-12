import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from lang2vec.lang2vec import get_features, available_languages
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.info

# Setup cache directory
cache = Path("lang2vec_cache")
cache.mkdir(exist_ok=True)

# Define language codes
codes = """kat_Geor yor_Latn lao_Laoo xho_Latn zul_Latn kan_Knda pan_Guru tso_Latn pbt_Arab
tel_Telu tgk_Cyrl urd_Latn nya_Latn sna_Latn swh_Latn fuv_Latn ibo_Latn jav_Latn
sin_Latn arb_Latn khm_Khmr ary_Arab bod_Tibt hat_Latn heb_Hebr kac_Latn khk_Cyrl
luo_Latn mri_Latn plt_Latn tsn_Latn wol_Latn grn_Latn guj_Gujr nso_Latn sin_Sinh
sun_Latn tam_Taml hau_Latn hin_Deva ilo_Latn kin_Latn ory_Orya snd_Arab ssw_Latn
eus_Latn shn_Mymr amh_Ethi gaz_Latn hye_Armn kaz_Cyrl lin_Latn sot_Latn war_Latn
ben_Beng ben_Latn hin_Latn mya_Mymr apc_Arab arz_Arab mal_Latn som_Latn arb_Arab
asm_Beng bam_Latn ckb_Arab kea_Latn mlt_Latn pes_Arab ars_Arab isl_Latn urd_Arab
azj_Latn ell_Grek lvs_Latn tha_Thai tir_Ethi als_Latn kir_Cyrl lug_Latn acm_Arab"""

# Extract language codes
available = set(available_languages())
langs = sorted({c.split('_')[0] for c in codes.split() if c.split('_')[0] in available})
sets = ("syntax_wals", "syntax_sswl")

# Helper to load or cache data
def load_or_make(fname, maker):
    path = cache / fname
    if path.exists():
        log(f"Loaded cached: {fname}")
        return pd.read_csv(path, index_col=0)
    log(f"Creating: {fname}")
    df = maker()
    df.to_csv(path)
    return df

# Fetch features
raw = {}
for s in sets:
    raw[s] = load_or_make(f"{s}_raw.csv", lambda s=s: pd.DataFrame({
        l: [np.nan if x in ("--", "") else float(x) for x in get_features([l], s)[l]]
        for l in langs
    }).T)

# Combine and clean data
vecs_raw = load_or_make("combined_raw.csv", lambda: pd.concat(raw.values(), axis=1))
vecs_raw = vecs_raw.loc[:, ~vecs_raw.isna().all()]
filtered = vecs_raw[vecs_raw.isna().mean(axis=1) < 0.5]
filtered = filtered.loc[:, filtered.isna().mean() < 0.5]
vecs = filtered.fillna(filtered.mean())

# Scale and reduce dimensions for visualization
scaled = StandardScaler().fit_transform(vecs)
embedding = TSNE(n_components=2, random_state=42).fit_transform(scaled)

# Silhouette analysis to find best k
silhouette_scores = []
cluster_range = range(5, 25)

for k in cluster_range:
    clustering = AgglomerativeClustering(n_clusters=k)
    labels_k = clustering.fit_predict(scaled)
    score = silhouette_score(scaled, labels_k)
    silhouette_scores.append(score)
    print(f"Silhouette score for {k} clusters: {score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different Cluster Counts")
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_scores.png", dpi=300)
plt.show()

# Choose best cluster count (manual or from max score)
best_k = cluster_range[np.argmax(silhouette_scores)]
final_clustering = AgglomerativeClustering(n_clusters=best_k)
final_labels = final_clustering.fit_predict(scaled)

# Visualize language clusters
plt.figure(figsize=(10, 6))
for i in range(best_k):
    idx = final_labels == i
    plt.scatter(embedding[idx, 0], embedding[idx, 1], label=f"Cluster {i}", alpha=0.7)

for i, lang in enumerate(vecs.index):
    plt.text(embedding[i, 0], embedding[i, 1], lang, fontsize=8, alpha=0.6)

plt.title(f"Language Clusters (k={best_k}) Based on Syntactic Features")
plt.legend()
plt.tight_layout()
plt.savefig("language_clusters.png", dpi=300)
plt.show()

# Print languages in each cluster
clustered_langs = pd.Series(final_labels, index=vecs.index)
for cluster_id in sorted(clustered_langs.unique()):
    langs_in_cluster = clustered_langs[clustered_langs == cluster_id].index.tolist()
    print(f"\nCluster {cluster_id} ({len(langs_in_cluster)} languages):")
    print(", ".join(langs_in_cluster))
