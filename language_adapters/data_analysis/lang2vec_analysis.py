import numpy as np
import pandas as pd
from pathlib import Path
from lang2vec.lang2vec import get_features, available_languages
from sklearn.metrics.pairwise import cosine_similarity

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.info

cache = Path("lang2vec_cache")
cache.mkdir(exist_ok=True)

codes = """kat_Geor yor_Latn lao_Laoo xho_Latn zul_Latn kan_Knda pan_Guru tso_Latn pbt_Arab
tel_Telu tgk_Cyrl urd_Latn nya_Latn sna_Latn swh_Latn fuv_Latn ibo_Latn jav_Latn
sin_Latn arb_Latn khm_Khmr ary_Arab bod_Tibt hat_Latn heb_Hebr kac_Latn khk_Cyrl
luo_Latn mri_Latn plt_Latn tsn_Latn wol_Latn grn_Latn guj_Gujr nso_Latn sin_Sinh
sun_Latn tam_Taml hau_Latn hin_Deva ilo_Latn kin_Latn ory_Orya snd_Arab ssw_Latn
eus_Latn shn_Mymr amh_Ethi gaz_Latn hye_Armn kaz_Cyrl lin_Latn sot_Latn war_Latn
ben_Beng ben_Latn hin_Latn mya_Mymr apc_Arab arz_Arab mal_Latn som_Latn arb_Arab
asm_Beng bam_Latn ckb_Arab kea_Latn mlt_Latn pes_Arab ars_Arab isl_Latn urd_Arab
azj_Latn ell_Grek lvs_Latn tha_Thai tir_Ethi als_Latn kir_Cyrl lug_Latn acm_Arab"""
langs = sorted({c.split('_')[0] for c in codes.split() if c.split('_')[0] in available_languages()})
sets = ("syntax_wals", "syntax_sswl")

def load_or_make(fname, maker):
    path = cache / fname
    if path.exists():
        log(f"from precomputed: {fname}")
        return pd.read_csv(path, index_col=0)
    log(f"use cached data: {fname}")
    df = maker()
    df.to_csv(path)
    return df

raw = {}
for s in sets:
    raw[s] = load_or_make(f"{s}_raw.csv", lambda s=s: pd.DataFrame({
        l: [np.nan if x in ("--", "") else float(x) for x in get_features([l], s)[l]]
        for l in langs
    }).T)

# Combine, clean, impute
vecs_raw = load_or_make("combined_raw.csv", lambda: pd.concat(raw.values(), axis=1))
vecs_raw = vecs_raw.loc[:, ~vecs_raw.isna().all()]
vecs_imp = load_or_make("combined_imputed.csv", lambda: vecs_raw.fillna(vecs_raw.mean()))

# Compute similarity
sim_df = load_or_make("similarity_cosine.csv", lambda: pd.DataFrame(
    cosine_similarity(vecs_imp), index=langs, columns=langs))

pairs = sim_df.where(~np.eye(len(sim_df), dtype=bool)).stack()
log("\nTop 15 closest languages:\n" + str(pairs.nlargest(15)))
log("\nTop 15 most distant languages:\n" + str(pairs.nsmallest(15)))
