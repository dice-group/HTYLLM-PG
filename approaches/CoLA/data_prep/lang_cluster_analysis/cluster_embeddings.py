#!/usr/bin/env python3
"""Cluster FLORES embedding vectors with optional auto-k logic."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_language_spec(spec: Optional[str]) -> Optional[List[str]]:
    if not spec:
        return None
    path = Path(spec)
    if path.exists():
        langs = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return langs or None
    return [token.strip() for token in spec.split(",") if token.strip()]


def load_embeddings(csv_path: Path, languages: Optional[Sequence[str]] = None):
    df = pd.read_csv(csv_path, low_memory=False)
    if languages:
        df = df[df["code"].isin(languages)]
    feature_cols = [c for c in df.columns if c.startswith("llm_emb_")]
    if not feature_cols:
        raise ValueError(f"Cannot find embedding columns in {csv_path}")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    langs = df["code"].tolist()
    families = df["family"].tolist()
    return langs, families, X


def cluster_data(X: np.ndarray, method: str, n_clusters: int, random_state: int = 42) -> np.ndarray:
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown method {method}")
    labels = model.fit_predict(X)
    return labels


def auto_determine_k(X: np.ndarray, method: str, min_k: int, max_k: int) -> Tuple[int, List[Tuple[int, float]]]:
    best_k = min_k
    best_score = -1.0
    scores = []
    for k in range(min_k, min(max_k, len(X) - 1) + 1):
        labels = cluster_data(X, method, k)
        counts = pd.Series(labels).value_counts()
        if len(set(labels)) < 2 or (counts < 2).any():
            continue
        score = silhouette_score(X, labels)
        scores.append((k, score))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores


def save_clusters(langs: Sequence[str], labels: Iterable[int], out_path: Path):
    data = {lang: int(label) for lang, label in zip(langs, labels)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    else:
        df = pd.DataFrame([{"code": lang, "cluster": label} for lang, label in data.items()])
        df.to_csv(out_path, index=False)


def plot_silhouette_curve(scores: List[Tuple[int, float]], output_path: Path):
    if not scores:
        return
    ks, vals = zip(*scores)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette analysis")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_clusters(coords: np.ndarray, langs: Sequence[str], families: Sequence[str], labels: Sequence[int], output_path: Path, color_by: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    unique_items = sorted(set(labels if color_by == "cluster" else families))
    mapper = {item: idx for idx, item in enumerate(unique_items)}
    colors = [mapper[item] for item in (labels if color_by == "cluster" else families)]
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="tab20", s=40, alpha=0.8, edgecolor="k", linewidth=0.2)
    for (x, y), lang, fam, cluster in zip(coords, langs, families, labels):
        plt.text(x, y, f"{lang} ({fam})", fontsize=6, ha="center", va="center", alpha=0.7)
    legend_labels = unique_items[:15]
    handles = []
    for item in legend_labels:
        handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(mapper[item])), label=str(item))
        )
    if len(unique_items) > len(legend_labels):
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color="gray", label="(others)"))
    if handles:
        plt.legend(handles=handles, title=color_by.capitalize(), bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.title(f"Embedding projection colored by {color_by}")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_top_languages(langs: Sequence[str], families: Sequence[str], labels: Sequence[int], features: np.ndarray, top_n: int, output_path: Path):
    if top_n <= 0:
        return []
    if (pd.Series(labels).value_counts() < 2).any():
        print("Warning: some clusters have <2 members; skipping top-N filtering")
        return []
    sil_vals = silhouette_samples(features, labels)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster FLORES embeddings with optional auto-k")
    parser.add_argument("--input", required=True, help="Path to embedding CSV from embed_flores_langs.py")
    parser.add_argument("--output", required=True, help="Where to save cluster assignments (json/csv)")
    parser.add_argument("--method", choices=["kmeans", "agglomerative"], default="kmeans")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters when auto-k is disabled")
    parser.add_argument("--auto-k", action="store_true", help="Enable silhouette-based automatic k selection")
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--languages", help="Comma-separated list or file with language codes to include")
    parser.add_argument("--scale", action="store_true", help="Apply StandardScaler before clustering (recommended for kmeans)")
    parser.add_argument("--plots-dir", help="Directory to save diagnostic plots (silhouette curve, 2D projections)")
    parser.add_argument("--top-n", type=int, help="Retain only the top-N languages with highest silhouette scores")
    parser.add_argument("--top-n-output", help="Where to save the filtered top-N mapping (defaults to <output>_top.json if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
    langs, families, X = load_embeddings(input_path, parse_language_spec(args.languages))
    if len(langs) < max(args.min_k, 2):
        raise SystemExit("Need at least min_k languages to cluster")

    if args.scale or args.method == "kmeans":
        X = StandardScaler().fit_transform(X)

    if args.auto_k:
        k, scores = auto_determine_k(X, args.method, args.min_k, args.max_k)
        print(f"Auto-selected k={k} based on silhouette score")
        if plots_dir:
            plot_silhouette_curve(scores, plots_dir / f"{Path(args.output).stem}_silhouette.png")
    else:
        k = args.clusters
    labels = cluster_data(X, args.method, k)
    output_path = Path(args.output)
    save_clusters(langs, labels, output_path)
    counts = pd.Series(labels).value_counts().sort_index()
    print(f"Saved clusters to {args.output} (k={k})")
    print("Cluster sizes:")
    print(counts)

    if args.top_n:
        top_path = Path(args.top_n_output) if args.top_n_output else output_path.with_name(output_path.stem + "_top.json")
        top_data = select_top_languages(langs, families, labels, X, args.top_n, top_path)
        if top_data:
            print(f"Top {len(top_data)} languages saved to {top_path}")

    if plots_dir:
        coords = PCA(n_components=2, random_state=0).fit_transform(X)
        base = Path(args.output).stem
        plot_clusters(coords, langs, families, labels, plots_dir / f"{base}_by_cluster.png", color_by="cluster")
        plot_clusters(coords, langs, families, labels, plots_dir / f"{base}_by_family.png", color_by="family")


if __name__ == "__main__":
    main()
