#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples, silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PREP_DIR = SCRIPT_DIR.parent
LANG_LIST_PATH = DATA_PREP_DIR / "processed_artifacts" / "filtered_languages.csv"
LANG2VEC_DIR = SCRIPT_DIR / "lang2vec"

sys.path.insert(0, str(LANG2VEC_DIR))
from lang2vec import lang2vec as l2v  # noqa: E402


def parse_language_spec(spec: Optional[str]) -> Optional[List[str]]:
    if not spec:
        return None
    path = Path(spec)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [token.strip() for token in spec.split(",") if token.strip()]


def load_langs(languages: Optional[List[str]] = None):
    df = pd.read_csv(LANG_LIST_PATH)[["code", "family"]]
    if languages:
        df = df[df["code"].isin(languages)]
    overlap = sorted(set(df["code"]) & set(l2v.DISTANCE_LANGUAGES))
    fam_map = dict(zip(df["code"], df["family"]))
    families = [fam_map[lang] for lang in overlap]
    return overlap, families


def build_distance_matrix(langs, distance_type):
    matrix = np.asarray(l2v.distance(distance_type, langs), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def cluster_with_k(matrix, k: int, linkage: str):
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=linkage)
    return model.fit_predict(matrix)


def auto_k(matrix, min_k: int, max_k: int, method: str) -> Tuple[int, List[Tuple[int, float]]]:
    best_k = min_k
    best_score = -1.0
    scores = []
    for k in range(min_k, min(max_k, matrix.shape[0] - 1) + 1):
        labels = cluster_with_k(matrix, k, method)
        counts = pd.Series(labels).value_counts()
        if len(set(labels)) < 2 or (counts < 2).any():
            continue
        score = silhouette_score(matrix, labels, metric="precomputed")
        scores.append((k, score))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores


def save_clusters(langs, labels, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {lang: int(label) for lang, label in zip(langs, labels)}
    if output_path.suffix == ".json":
        output_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    else:
        pd.DataFrame([{"code": lang, "cluster": lab} for lang, lab in mapping.items()]).to_csv(
            output_path, index=False
        )


def plot_silhouette(scores: List[Tuple[int, float]], output_path: Path):
    if not scores:
        return
    ks, vals = zip(*scores)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, vals, marker="o")
    plt.title("Lang2Vec silhouette analysis")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_distance_projection(
    matrix: np.ndarray,
    langs: Sequence[str],
    families: Sequence[str],
    labels: Sequence[int],
    output_path: Path,
    color_by: str,
):
    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        normalized_stress="auto",
    ).fit_transform(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    items = sorted(set(labels if color_by == "cluster" else families))
    mapper = {item: idx for idx, item in enumerate(items)}
    colors = [mapper[item] for item in (labels if color_by == "cluster" else families)]
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="tab20", s=45, alpha=0.85, edgecolor="k", linewidth=0.2)
    for (x, y), lang, fam in zip(coords, langs, families):
        plt.text(x, y, f"{lang} ({fam})", fontsize=6, ha="center", va="center", alpha=0.7)
    legend_items = items[:15]
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(mapper[item])), label=str(item))
        for item in legend_items
    ]
    if len(items) > len(legend_items):
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color="gray", label="(others)"))
    if handles:
        plt.legend(handles=handles, title=color_by.capitalize(), bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.title(f"Lang2Vec {color_by} visualization")
    plt.xlabel("MDS dim 1")
    plt.ylabel("MDS dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster languages with Lang2Vec distances.")
    parser.add_argument("--distance-type", default="genetic", choices=l2v.DISTANCES)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--method", default="average")
    parser.add_argument("--languages", help="Comma list or file with language codes", default=None)
    parser.add_argument("--output", required=True, help="Where to store mapping (json/csv)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--plots-dir", help="Directory to store silhouette and projection plots")
    parser.add_argument("--top-n", type=int, help="Retain only the top-N languages with highest silhouette scores")
    parser.add_argument(
        "--top-n-output",
        help="Where to store the filtered mapping (defaults to <output>_top.json)",
    )
    return parser.parse_args()


def select_top_languages(langs, families, labels, matrix, top_n, output_path: Path):
    if top_n <= 0:
        return []
    if (pd.Series(labels).value_counts() < 2).any():
        print("Warning: some clusters have <2 members; skipping top-N filtering")
        return []
    sil_vals = silhouette_samples(matrix, labels, metric="precomputed")
    order = np.argsort(-sil_vals)[: min(top_n, len(langs))]
    selection = [
        {
            "code": langs[i],
            "family": families[i],
            "cluster": int(labels[i]),
            "silhouette": float(sil_vals[i]),
        }
        for i in order
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def main():
    args = parse_args()
    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
    langs, families = load_langs(parse_language_spec(args.languages))
    if args.limit:
        langs = langs[: args.limit]
        families = families[: args.limit]

    if len(langs) < max(args.min_k, 2):
        raise SystemExit("Need more languages to cluster")

    matrix = build_distance_matrix(langs, args.distance_type)
    if args.auto_k:
        k, scores = auto_k(matrix, args.min_k, args.max_k, args.method)
        print(f"Auto-selected k={k} for Lang2Vec distances")
        if plots_dir:
            plot_silhouette(scores, plots_dir / f"{Path(args.output).stem}_silhouette.png")
    else:
        k = args.clusters
    labels = cluster_with_k(matrix, k, args.method)
    output_path = Path(args.output)
    save_clusters(langs, labels, output_path)
    print(f"Saved clusters to {args.output} (k={k})")
    if args.top_n:
        top_path = Path(args.top_n_output) if args.top_n_output else output_path.with_name(output_path.stem + "_top.json")
        subset = select_top_languages(langs, families, labels, matrix, args.top_n, top_path)
        if subset:
            print(f"Top {len(subset)} languages saved to {top_path}")
    if plots_dir:
        base = Path(args.output).stem
        plot_distance_projection(matrix, langs, families, labels, plots_dir / f"{base}_by_cluster.png", "cluster")
        plot_distance_projection(matrix, langs, families, labels, plots_dir / f"{base}_by_family.png", "family")


if __name__ == "__main__":
    main()
