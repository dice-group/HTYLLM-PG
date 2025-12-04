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


def auto_k(matrix, min_k: int, max_k: int, method: str, min_cluster_size: int = 2) -> Tuple[int, List[Tuple[int, float]]]:
    best_k = min_k
    best_score = -1.0
    scores = []
    for k in range(min_k, min(max_k, matrix.shape[0] - 1) + 1):
        labels = cluster_with_k(matrix, k, method)
        counts = pd.Series(labels).value_counts()
        if len(set(labels)) < 2:
            continue
        if min_cluster_size > 1 and (counts < min_cluster_size).any():
            continue
        try:
            score = silhouette_score(matrix, labels, metric="precomputed")
        except ValueError:
            continue
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


def plot_distance_projection(matrix: np.ndarray, langs: Sequence[str], families: Sequence[str], labels: Sequence[int], output_path: Path, color_by: str):
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
    parser.add_argument("--top-n-output", help="Where to store the filtered mapping (defaults to <output>_top.json)")
    parser.add_argument("--top-per-cluster", type=int, default=0, help="Pick this many languages per cluster ranked by silhouette")
    parser.add_argument("--top-per-cluster-output", help="Where to store the balanced subset (defaults to <output>_top_per_cluster.json)")
    parser.add_argument("--best-clusters", type=int, default=0, help="Limit balanced selection to the best-performing clusters by average silhouette")
    parser.add_argument("--low-perf-csv", help="CSV with per-language accuracies (e.g., Belebele) to prioritize lower-performing languages.")
    parser.add_argument("--low-perf-threshold", type=float, default=0.0, help="Accuracy threshold used when --low-perf-csv is provided (<= threshold preferred).")
    parser.add_argument("--low-perf-per-cluster", type=int, default=0, help="Number of languages per cluster to emit after low-performance filtering.")
    parser.add_argument("--low-perf-output", help="Where to store the filtered mapping (defaults to <output>_lowperf.json).")
    parser.add_argument("--low-perf-report", help="Optional JSON report describing candidate accuracies per cluster.")
    parser.add_argument("--low-perf-plot", help="Optional highlight plot path to visualize selected languages.")
    return parser.parse_args()


def compute_silhouettes(matrix: np.ndarray, labels: Sequence[int]):
    counts = pd.Series(labels).value_counts()
    labels = np.asarray(labels)
    valid_mask = np.array([counts.get(label, 0) >= 2 for label in labels], dtype=bool)
    if not valid_mask.any():
        print("Warning: no clusters with >=2 members; silhouette metrics unavailable")
        return None
    if valid_mask.all():
        return silhouette_samples(matrix, labels, metric="precomputed")
    print("Warning: some clusters have <2 members; assigning zero silhouette scores to those languages")
    sub_matrix = matrix[np.ix_(valid_mask, valid_mask)]
    sub_labels = labels[valid_mask]
    sub_scores = silhouette_samples(sub_matrix, sub_labels, metric="precomputed")
    sil_vals = np.zeros(len(labels), dtype=float)
    sil_vals[valid_mask] = sub_scores
    return sil_vals


def select_top_languages(langs, families, labels, sil_vals, top_n, output_path: Path):
    if top_n <= 0 or sil_vals is None:
        return []
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


def select_top_per_cluster(langs, families, labels, sil_vals, per_cluster: int, output_path: Path, best_clusters: int = 0):
    if per_cluster <= 0 or sil_vals is None:
        return []
    counts = pd.Series(labels).value_counts()
    low_clusters = counts[counts < per_cluster]
    stats = (
        pd.DataFrame({"cluster": labels, "silhouette": sil_vals})
        .groupby("cluster")
        .agg(count=("silhouette", "size"), avg_sil=("silhouette", "mean"))
    )
    valid_clusters = stats[stats["count"] >= per_cluster].sort_values("avg_sil", ascending=False)
    if valid_clusters.empty:
        print("Warning: no clusters have enough members for balanced selection")
        return []
    cluster_order = valid_clusters.index.tolist()
    if best_clusters > 0:
        cluster_order = cluster_order[:best_clusters]
        if len(cluster_order) < best_clusters:
            print(f"Warning: only {len(cluster_order)} clusters satisfy the size constraint (requested {best_clusters})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection = []
    for cluster_id in cluster_order:
        member_indices = [i for i, lab in enumerate(labels) if lab == cluster_id]
        member_indices.sort(key=lambda idx: sil_vals[idx], reverse=True)
        for rank, idx in enumerate(member_indices[:per_cluster], start=1):
            selection.append(
                {
                    "code": langs[idx],
                    "family": families[idx],
                    "cluster": int(labels[idx]),
                    "silhouette": float(sil_vals[idx]),
                    "cluster_rank": rank,
                }
            )
    output_path.write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def load_performance_table(csv_path: Path):
    if not csv_path or not csv_path.exists():
        print(f"Warning: accuracy CSV {csv_path} not found; skipping low-performance filtering")
        return {}
    df = pd.read_csv(csv_path)
    if "language" not in df.columns or "acc" not in df.columns:
        print(f"Warning: accuracy CSV {csv_path} missing required columns 'language' and 'acc'")
        return {}
    df = df.dropna(subset=["language", "acc"])
    return dict(zip(df["language"], df["acc"]))


def build_code_subset_lookup():
    df = pd.read_csv(LANG_LIST_PATH)[["code", "subset"]].dropna()
    mapping = {}
    for _, row in df.iterrows():
        mapping.setdefault(row["code"], set()).add(row["subset"])
    return mapping


def pick_accuracy_for_code(code, acc_map, code2subset):
    subsets = code2subset.get(code, [])
    scores = [(subset, acc_map[subset]) for subset in subsets if subset in acc_map]
    if not scores:
        return None
    scores.sort(key=lambda item: item[1])
    return scores[0]


def filter_low_performance_selection(candidates, per_cluster: int, threshold: float, acc_csv: Path, output_path: Path, report_path: Optional[Path]):
    if per_cluster <= 0:
        return []
    acc_map = load_performance_table(acc_csv)
    code2subset = build_code_subset_lookup()
    clusters = {}
    for entry in candidates:
        clusters.setdefault(entry["cluster"], []).append(dict(entry))
    for entries in clusters.values():
        entries.sort(key=lambda item: item.get("cluster_rank", 0))

    filtered = []
    report = {"threshold": threshold, "clusters": {}}
    for cluster_id in sorted(clusters):
        entries = clusters[cluster_id]
        low_perf = []
        fallback = []
        report["clusters"][cluster_id] = {"candidates": []}
        for entry in entries:
            code = entry["code"]
            acc_pair = pick_accuracy_for_code(code, acc_map, code2subset) if acc_map else None
            acc = acc_pair[1] if acc_pair else None
            subset = acc_pair[0] if acc_pair else None
            if subset is None:
                subset_candidates = sorted(code2subset.get(code, []))
                subset = subset_candidates[0] if subset_candidates else None
            entry["accuracy"] = acc
            entry["subset"] = subset
            if acc is not None and threshold > 0 and acc <= threshold:
                low_perf.append(entry)
            elif acc is None and threshold > 0:
                fallback.append(entry)
            elif threshold <= 0:
                low_perf.append(entry)
            else:
                fallback.append(entry)
            report["clusters"][cluster_id]["candidates"].append(entry)

        chosen = []
        while low_perf and len(chosen) < per_cluster:
            chosen.append(low_perf.pop(0))
        while fallback and len(chosen) < per_cluster:
            chosen.append(fallback.pop(0))
        if len(chosen) < per_cluster:
            print(f"Warning: only {len(chosen)} candidates found for cluster {cluster_id} (requested {per_cluster})")
        report["clusters"][cluster_id]["selected"] = chosen
        filtered.extend(chosen)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered, indent=2) + "\n")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return filtered


def plot_selection_highlight(matrix, langs, labels, selection, output_path: Path):
    if not selection or not output_path:
        return
    selection_map = {entry["code"]: entry for entry in selection}
    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        normalized_stress="auto",
    ).fit_transform(matrix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=30, alpha=0.2, linewidth=0)

    sel_indices = [i for i, lang in enumerate(langs) if lang in selection_map]
    if not sel_indices:
        print("Warning: no selected languages to highlight")
        plt.close()
        return
    sel_x = coords[sel_indices, 0]
    sel_y = coords[sel_indices, 1]
    plt.scatter(
        sel_x,
        sel_y,
        facecolors="none",
        edgecolors="red",
        s=220,
        linewidth=1.6,
        label="Selected languages",
    )
    for idx in sel_indices:
        lang = langs[idx]
        payload = selection_map[lang]
        label = payload.get("subset") or lang
        acc = payload.get("accuracy")
        if acc is not None:
            label += f" ({acc:.2f})"
        plt.text(
            coords[idx, 0],
            coords[idx, 1],
            label,
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0),
        )
    plt.title("Lang2Vec projection with highlighted selections")
    plt.xlabel("MDS dim 1")
    plt.ylabel("MDS dim 2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"Highlight plot saved to {output_path}")


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
    sil_vals = None
    if args.top_n or args.top_per_cluster:
        sil_vals = compute_silhouettes(matrix, labels)
    top_per_cluster_selection = []
    if args.top_n:
        top_path = Path(args.top_n_output) if args.top_n_output else output_path.with_name(output_path.stem + "_top.json")
        subset = select_top_languages(langs, families, labels, sil_vals, args.top_n, top_path)
        if subset:
            print(f"Top {len(subset)} languages saved to {top_path}")
    if args.top_per_cluster:
        balanced_path = (
            Path(args.top_per_cluster_output)
            if args.top_per_cluster_output
            else output_path.with_name(output_path.stem + "_top_per_cluster.json")
        )
        top_per_cluster_selection = select_top_per_cluster(
            langs,
            families,
            labels,
            sil_vals,
            args.top_per_cluster,
            balanced_path,
            best_clusters=args.best_clusters,
        )
        if top_per_cluster_selection:
            total = len(top_per_cluster_selection)
            print(f"Balanced top-per-cluster selection ({total} langs) saved to {balanced_path}")
    if plots_dir:
        base = Path(args.output).stem
        plot_distance_projection(matrix, langs, families, labels, plots_dir / f"{base}_by_cluster.png", "cluster")
        plot_distance_projection(matrix, langs, families, labels, plots_dir / f"{base}_by_family.png", "family")

    low_perf_selection = []
    if args.low_perf_csv and args.low_perf_per_cluster:
        perf_output = Path(args.low_perf_output) if args.low_perf_output else output_path.with_name(output_path.stem + "_lowperf.json")
        perf_report = Path(args.low_perf_report) if args.low_perf_report else None
        if not top_per_cluster_selection:
            print("Warning: --low-perf-* flags require --top-per-cluster to be set; skipping low-performance filtering")
        else:
            low_perf_selection = filter_low_performance_selection(
                top_per_cluster_selection,
                args.low_perf_per_cluster,
                args.low_perf_threshold,
                Path(args.low_perf_csv),
                perf_output,
                perf_report,
            )
            if low_perf_selection:
                print(f"Low-performance filtered selection saved to {perf_output}")
    plot_path = None
    if args.low_perf_plot:
        plot_path = Path(args.low_perf_plot)
    elif plots_dir and low_perf_selection:
        plot_path = plots_dir / f"{output_path.stem}_selection.png"
    if plot_path and low_perf_selection:
        plot_selection_highlight(matrix, langs, labels, low_perf_selection, plot_path)


if __name__ == "__main__":
    main()
