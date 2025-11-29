import sys
import numpy as np
import pandas as pd

from collections import OrderedDict
from pathlib import Path

"""
Quick Lang2Vec coverage report for the FLORES languages we already filtered.
"""
_np_load = np.load


def _np_load_allow_pickle(*args, **kwargs):
    kwargs.setdefault("allow_pickle", True)
    return _np_load(*args, **kwargs)


np.load = _np_load_allow_pickle


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PREP_DIR = SCRIPT_DIR
LANG_LIST_PATH = DATA_PREP_DIR / "processed_artifacts" / "filtered_languages.csv"
LANG2VEC_DIR = DATA_PREP_DIR / "lang2vec"

# Make sure we import the local lang2vec package that ships with this repo.
sys.path.insert(0, str(LANG2VEC_DIR))
from lang2vec import lang2vec as l2v


def load_languages():
    df = pd.read_csv(LANG_LIST_PATH)
    return sorted(df["code"].unique())


def summarize_feature_set(langs, feature_key):
    try:
        feature_payload = l2v.get_features(langs, feature_key, header=True)
    except Exception as exc:  # pragma: no cover - informational
        return {
            "feature_key": feature_key,
            "feature_dim": 0,
            "covered_langs": [],
            "coverage_ratio": 0.0,
            "error": str(exc),
        }
    feature_names = feature_payload.pop("CODE")
    available_langs = []
    for lang, values in feature_payload.items():
        if any(v != "--" for v in values):
            available_langs.append(lang)
    return {
        "feature_key": feature_key,
        "feature_dim": len(feature_names),
        "covered_langs": available_langs,
        "coverage_ratio": len(available_langs) / len(langs) if langs else 0.0,
        "error": "",
    }


def main():
    all_langs = load_languages()
    lang2vec_codes = set(l2v.LETTER_CODES.keys()) | set(l2v.LETTER_CODES.values())
    covered = sorted(set(all_langs) & lang2vec_codes)
    missing = sorted(set(all_langs) - lang2vec_codes)

    print(f"Total FLORES languages in CSV: {len(all_langs)}")
    print(f"Lang2Vec coverage: {len(covered)} ({len(covered) / len(all_langs):.1%})")
    if missing:
        print(f"Sample missing codes: {', '.join(missing[:10])} ...")

    distance_langs = set(l2v.DISTANCE_LANGUAGES)
    dist_overlap = sorted(set(all_langs) & distance_langs)
    dist_missing = sorted(set(all_langs) - distance_langs)
    print(f"Similarity/distances available for {len(dist_overlap)}/{len(all_langs)} langs ({len(dist_overlap)/len(all_langs):.1%}).")
    if dist_missing:
        print(f"Sample without distance metadata: {', '.join(dist_missing[:10])} ...")

    feature_summary = OrderedDict()
    for feature_key in ["syntax_wals", "phonology_wals", "inventory_phoible_aa", "geo", "fam"]:
        if not covered:
            break
        summary = summarize_feature_set(covered, feature_key)
        feature_summary[feature_key] = summary

    print("\nSelected feature coverage:")
    for key, info in feature_summary.items():
        if info["error"]:
            print(f"- {key:20s} unsupported -> {info['error'].split('.')[0]}")
            continue
        pct = info["coverage_ratio"] * 100
        print(f"- {key:20s} {len(info['covered_langs']):4d}/{len(covered):4d} langs ({pct:5.1f}%), dim={info['feature_dim']}")

    print("\nAnalysis tips:")
    print("- Use lang2vec.get_features(<langs>, 'syntax_wals+phonology_wals') to build typology matrices.")
    print("- Combine the coverage info above with our FLORES embeddings to cluster languages or control for family/geo signals.")
    print("- For languages with distance coverage, call lang2vec.distance('genetic', ['eng', 'deu', ...]) to fetch similarity scores.")


if __name__ == "__main__":
    main()
