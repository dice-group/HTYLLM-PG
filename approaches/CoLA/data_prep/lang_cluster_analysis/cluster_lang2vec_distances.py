import argparse
import sys
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

"""
Cluster FLORES languages using Lang2Vec distance matrices.

Example:
    python cluster_lang2vec_distances.py --distance-type genetic --clusters 12
"""

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PREP_DIR = SCRIPT_DIR.parent
LANG_LIST_PATH = DATA_PREP_DIR / "processed_artifacts" / "filtered_languages.csv"
LANG2VEC_DIR = SCRIPT_DIR / "lang2vec"

sys.path.insert(0, str(LANG2VEC_DIR))
from lang2vec import lang2vec as l2v


def load_langs():
    df = pd.read_csv(LANG_LIST_PATH)
    return sorted(df["code"].unique())


def build_distance_matrix(langs, distance_type):
    matrix = np.asarray(l2v.distance(distance_type, langs), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def cluster_languages(matrix, langs, n_clusters, method):
    condensed = squareform(matrix, checks=False)
    linkage_matrix = linkage(condensed, method=method)
    labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    groups = defaultdict(list)
    for lang, label in zip(langs, labels):
        groups[label].append(lang)
    return linkage_matrix, groups


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster languages with Lang2Vec distances.")
    parser.add_argument("--distance-type", default="genetic", choices=l2v.DISTANCES, help="Distance space to use.")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters for fcluster.")
    parser.add_argument("--method", default="average", help="Linkage method (scipy.cluster.hierarchy.linkage).")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of languages (for quick tests).")
    return parser.parse_args()


def main():
    args = parse_args()
    all_langs = load_langs()
    overlap = sorted(set(all_langs) & set(l2v.DISTANCE_LANGUAGES))
    if args.limit:
        overlap = overlap[: args.limit]

    if len(overlap) < 2:
        raise SystemExit("Need at least two languages with distance coverage.")

    print(f"Clustering {len(overlap)} languages using '{args.distance_type}' distances...")
    dist_matrix = build_distance_matrix(overlap, args.distance_type)
    _, clusters = cluster_languages(dist_matrix, overlap, args.clusters, args.method)

    for cid in sorted(clusters):
        members = ", ".join(clusters[cid])
        print(f"Cluster {cid:02d} ({len(clusters[cid]):2d} langs): {members}")


if __name__ == "__main__":
    main()
