#!/usr/bin/env python3
"""
Visualize FLORES languages in 2D using Lang2Vec distances and family colors.

Example:
    python plot_lang2vec_clusters.py --distance-type genetic --clusters 15
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PREP_DIR = SCRIPT_DIR.parent
LANG_LIST_PATH = DATA_PREP_DIR / "processed_artifacts" / "filtered_languages.csv"
LANG2VEC_DIR = SCRIPT_DIR / "lang2vec"

sys.path.insert(0, str(LANG2VEC_DIR))
from lang2vec import lang2vec as l2v


def load_languages():
    df = pd.read_csv(LANG_LIST_PATH)
    df = df[["code", "family"]].drop_duplicates()
    return df.set_index("code")


def compute_coords(langs, distance_type):
    matrix = np.asarray(l2v.distance(distance_type, langs), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(matrix)
    return StandardScaler().fit_transform(coords)


def cluster_assignments(langs, distance_type, n_clusters):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = np.asarray(l2v.distance(distance_type, langs), dtype=float)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    return fcluster(linkage_matrix, n_clusters, criterion="maxclust")


def parse_args():
    parser = argparse.ArgumentParser(description="2D Lang2Vec plot with family colors.")
    parser.add_argument("--distance-type", default="genetic", choices=l2v.DISTANCES, help="Distance space.")
    parser.add_argument("--clusters", type=int, default=15, help="Number of clusters to print (hierarchical).")
    parser.add_argument("--limit", type=int, default=0, help="Optionally limit the language count.")
    parser.add_argument("--output", default="lang2vec_clusters.png", help="PNG output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    lang_df = load_languages()
    available = sorted(set(lang_df.index) & set(l2v.DISTANCE_LANGUAGES))
    if args.limit:
        available = available[: args.limit]
    if len(available) < 3:
        raise SystemExit("Need at least 3 languages for visualization.")

    coords = compute_coords(available, args.distance_type)
    families = [lang_df.loc[lang, "family"] for lang in available]
    uniq_families = sorted(set(families))
    family_to_id = {fam: idx for idx, fam in enumerate(uniq_families)}
    color_ids = [family_to_id[fam] for fam in families]

    # Optional logging of clusters using distances (not shown on plot)
    cluster_ids = cluster_assignments(available, args.distance_type, args.clusters)
    print(f"Computed clusters with {args.distance_type} distances:")
    for cid in sorted(set(cluster_ids)):
        members = [lang for lang, lid in zip(available, cluster_ids) if lid == cid]
        print(f"  Cluster {cid:02d} ({len(members)} langs): {', '.join(members[:15])}{' ...' if len(members)>15 else ''}")

    cmap = plt.get_cmap("tab20", max(len(uniq_families), 1))
    plt.figure(figsize=(12, 9))
    plt.scatter(coords[:, 0], coords[:, 1], c=color_ids, cmap=cmap, s=45, alpha=0.85, edgecolor="k", linewidth=0.2)
    for lang, (x, y) in zip(available, coords):
        plt.text(x, y, lang, fontsize=7, ha="center", va="center", alpha=0.7)

    handles = []
    max_legend = 15
    for fam in uniq_families[:max_legend]:
        handles.append(Line2D([0], [0], marker="o", color="w", label=fam, markerfacecolor=cmap(family_to_id[fam]), markersize=8))
    if len(uniq_families) > max_legend:
        handles.append(Line2D([0], [0], marker="o", color="w", label="(others)", markerfacecolor="gray", markersize=8))
    if handles:
        plt.legend(handles=handles, title="Family (top)", loc="upper right", fontsize=8)

    plt.title(f"Lang2Vec {args.distance_type} distances w/ family colors")
    plt.xlabel("MDS dimension 1")
    plt.ylabel("MDS dimension 2")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved 2D plot to {args.output}")


if __name__ == "__main__":
    main()
