import itertools, pathlib, urllib.request, zipfile, io, os, math, json
import numpy as np, pandas as pd
from langcodes import Language
import fasttext
from sklearn.metrics.pairwise import cosine_similarity

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

def structural_distance(tag_a, tag_b):
    """0.0 = identical; increasing numbers = more distant."""
    return Language.get(tag_a).distance(Language.get(tag_b))

struct_mat = pd.DataFrame(index=tags, columns=tags, dtype=float)
for a, b in itertools.product(tags, repeat=2):
    struct_mat.at[a, b] = structural_distance(a, b)


model_path = 'lid.176.bin'
if not pathlib.Path(model_path).exists():
    url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
    print('Downloading FastText model …')
    urllib.request.urlretrieve(url, model_path)

ft = fasttext.load_model(model_path)

def fasttext_vector(tag):
    """Get the 300-d vector for a language label (ISO-639-3)."""
    iso639_3 = tag.split('-')[0]          # keep part before hyphen
    return ft.get_word_vector(f'__label__{iso639_3}')

vecs = np.vstack([fasttext_vector(t) for t in tags])
cos_sim = cosine_similarity(vecs)
dist_mat = 1 - cos_sim                    # 0 = identical, 2 = opposite

ft_mat = pd.DataFrame(dist_mat, index=tags, columns=tags)

def top_pairs(df, k=150):
    out = []
    for i, j in itertools.combinations(range(len(df)), 2):
        out.append((df.iat[i, j], df.index[i], df.columns[j]))
    return sorted(out, key=lambda x: x[0])[:k]

print('\n 15 closest language pairs by STRUCTURAL distance')
for d, a, b in top_pairs(struct_mat):
    print(f'{a:10s} <-> {b:10s}  distance={d:.2f}')

print('\n15 closest language pairs by FastText (lexical) distance')
for d, a, b in top_pairs(ft_mat):
    print(f'{a:10s} <-> {b:10s}  distance={d:.3f}')
