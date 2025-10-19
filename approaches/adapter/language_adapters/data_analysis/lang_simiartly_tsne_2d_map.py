import itertools, pathlib, urllib.request
import numpy as np, pandas as pd
import fasttext
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

codes_raw = """
kat_Geor lao_Laoo xho_Latn zul_Latn pan_Guru tso_Latn pbt_Arab
tel_Telu tgk_Cyrl urd_Latn nya_Latn sna_Latn swh_Latn fuv_Latn
sin_Latn arb_Latn khm_Khmr ary_Arab bod_Tibt hat_Latn heb_Hebr kac_Latn
luo_Latn mri_Latn plt_Latn tsn_Latn wol_Latn grn_Latn guj_Gujr nso_Latn sin_Sinh
hau_Latn hin_Deva ilo_Latn kin_Latn ory_Orya snd_Arab ssw_Latn
eus_Latn shn_Mymr gaz_Latn hye_Armn lin_Latn 
sot_Latn war_Latn ben_Latn hin_Latn mya_Mymr som_Latn bam_Latn 
ckb_Arab kea_Latn mlt_Latn pes_Arab isl_Latn urd_Arab azj_Latn
ell_Grek lvs_Latn tha_Thai als_Latn kir_Cyrl lug_Latn
""".split()
tags = [c.replace('_', '-') for c in codes_raw]

model_path = "lid.176.bin"
if not pathlib.Path(model_path).exists():
    print("Downloading FastText model … (~126 MB)")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        model_path,
    )
ft = fasttext.load_model(model_path)

def ft_vec(tag):
    return ft.get_word_vector(f"__label__{tag.split('-')[0]}")

vecs = np.vstack([ft_vec(t) for t in tags])

n_clusters = 15
clust = AgglomerativeClustering(
    n_clusters=n_clusters, metric="cosine", linkage="average"
)
labels = clust.fit_predict(vecs)

emb = TSNE(
    n_components=2, metric="cosine", init="random", random_state=0
).fit_transform(vecs)

plt.figure(figsize=(11, 8))
for c in range(n_clusters):
    idx = labels == c
    plt.scatter(emb[idx, 0], emb[idx, 1], s=45, alpha=0.8, label=f"Cluster {c}")
    for x, y, tag in zip(emb[idx, 0], emb[idx, 1], np.array(tags)[idx]):
        plt.text(x, y, tag.split('-')[0], fontsize=7)

plt.title("Language similarity map (t-SNE on FastText embeddings)")
plt.axis("off")
plt.legend(fontsize=8, loc="upper right", frameon=False)
plt.tight_layout()
plt.savefig("language_similarity_tsne_map.png", dpi=300, bbox_inches="tight")
plt.show()
