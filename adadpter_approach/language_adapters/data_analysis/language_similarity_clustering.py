import itertools, pathlib, urllib.request, io, zipfile, os, math, json
import numpy as np, pandas as pd
from langcodes import Language
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

codes_raw = """
kat_Geor yor_Latn lao_Laoo xho_Latn zul_Latn kan_Knda pan_Guru tso_Latn pbt_Arab
tel_Telu tgk_Cyrl urd_Latn nya_Latn sna_Latn swh_Latn fuv_Latn ibo_Latn jav_Latn
sin_Latn arb_Latn khm_Khmr ary_Arab bod_Tibt hat_Latn heb_Hebr kac_Latn khk_Cyrl
luo_Latn mri_Latn plt_Latn tsn_Latn wol_Latn grn_Latn guj_Gujr nso_Latn sin_Sinh
sun_Latn tam_Taml hau_Latn hin_Deva ilo_Latn kin_Latn ory_Orya snd_Arab ssw_Latn
eus_Latn shn_Mymr amh_Ethi gaz_Latn hye_Armn kaz_Cyrl lin_Latn sot_Latn war_Latn
ben_Beng ben_Latn hin_Latn mya_Mymr apc_Arab arz_Arab mal_Mlym som_Latn arb_Arab
asm_Beng bam_Latn ckb_Arab kea_Latn mlt_Latn pes_Arab ars_Arab isl_Latn urd_Arab
azj_Latn ell_Grek lvs_Latn tha_Thai tir_Ethi als_Latn kir_Cyrl lug_Latn acm_Arab
""".split()
tags = [c.replace('_', '-') for c in codes_raw]

def structural_distance(a, b):
    return Language.get(a).distance(Language.get(b))

struct_mat = pd.DataFrame(index=tags, columns=tags, dtype=float)
for a, b in itertools.product(tags, repeat=2):
    struct_mat.at[a, b] = structural_distance(a, b)

model_path = "lid.176.bin"
if not pathlib.Path(model_path).exists():
    print("Downloading FastText model …")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        model_path,
    )
ft = fasttext.load_model(model_path)

def ft_vec(tag):
    return ft.get_word_vector(f"__label__{tag.split('-')[0]}")

vecs = np.vstack([ft_vec(t) for t in tags])
ft_dist = 1 - cosine_similarity(vecs)
ft_mat = pd.DataFrame(ft_dist, index=tags, columns=tags)

dist_mat = struct_mat.values

Z = linkage(dist_mat, method="average")
plt.figure(figsize=(10, 30))
dendrogram(Z, labels=tags, orientation="right", leaf_font_size=8)
plt.tight_layout()
plt.savefig("language_dendrogram.png", dpi=300, bbox_inches="tight")
plt.show()
