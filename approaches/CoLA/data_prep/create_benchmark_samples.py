#!/usr/bin/env python3
"""Cluster cached embeddings to build benchmark language subsets."""
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
import plotly.express as px
import plotly.graph_objects as go

PROCESSED_FILENAMES = {
    "metadata": "filtered_languages.csv",
    "onehot_embeddings": "onehot_embeddings.csv",
    "llm_embeddings": "llm_embeddings.csv",
    "onehot_tsne": "onehot_tsne.csv",
    "llm_tsne": "llm_tsne.csv",
}


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required artifact: {path}. Run preprocess_evaluable_languages.py first.")
    return path


def load_processed_artifacts(processed_dir):
    """Load cached metadata, embeddings, and t-SNE coordinates."""
    data = {}
    for key, filename in PROCESSED_FILENAMES.items():
        full_path = require_file(os.path.join(processed_dir, filename))
        data[key] = pd.read_csv(full_path)
    metadata = data["metadata"]
    artifacts = {
        "onehot": {"embeddings": data["onehot_embeddings"], "tsne": data["onehot_tsne"]},
        "llm": {"embeddings": data["llm_embeddings"], "tsne": data["llm_tsne"]},
    }
    return metadata, artifacts


def farthest_first_ordering(dist_matrix):
    """Greedy farthest-point selection."""
    n = dist_matrix.shape[0]
    start_idx = np.argmin(dist_matrix.sum(axis=1))
    ordered = [start_idx]
    remaining = set(range(n)) - {start_idx}
    while remaining:
        min_dists = {i: min(dist_matrix[i, sel] for sel in ordered) for i in remaining}
        next_idx = max(min_dists, key=min_dists.get)
        ordered.append(next_idx)
        remaining.remove(next_idx)
    return ordered


def select_optimal_sizes(
    embeddings_df,
    k_min=5,
    k_max=100,
    num_sizes=3,
    include_full_sample=False,
    total_langs=None,
):
    """Pick representative sample sizes using silhouette peaks."""
    print(f"Analyzing cluster quality for k={k_min}..{k_max}...")
    scores = []
    for k in range(k_min, k_max + 1, 2):
        if k >= len(embeddings_df):
            break
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=42)
        labels = kmeans.fit_predict(embeddings_df)
        scores.append((k, silhouette_score(embeddings_df, labels)))

    peaks = []
    for idx in range(1, len(scores) - 1):
        if scores[idx][1] > scores[idx - 1][1] and scores[idx][1] > scores[idx + 1][1]:
            peaks.append(scores[idx])
    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
    peak_ks = [p[0] for p in peaks_sorted]

    def clamp_k(value):
        if total_langs:
            value = min(value, total_langs - 1)
        return max(1, value)

    # Small tier: best silhouette peak (or fallback to k_min)
    if peak_ks:
        small_k = clamp_k(peak_ks[0])
    else:
        small_k = clamp_k(max(k_min, 10))

    # Medium tier: prefer peaks >= target range
    MEDIUM_MIN = 40
    MEDIUM_TARGET = 64
    medium_k = None
    for k in peak_ks:
        if k == small_k:
            continue
        if total_langs and k >= total_langs:
            continue
        if k >= MEDIUM_MIN:
            medium_k = clamp_k(k)
            break

    if medium_k is None:
        for k in peak_ks:
            if k != small_k and (not total_langs or k < total_langs):
                medium_k = clamp_k(k)
                break

    if medium_k is None:
        upper_bound = total_langs - 1 if total_langs else k_max
        target = clamp_k(int(round(MEDIUM_TARGET)))
        target = min(max(target, small_k + 1), upper_bound)
        medium_k = clamp_k(target)

    if medium_k <= small_k:
        medium_k = clamp_k(small_k + 1)

    selected = []
    if small_k > 0:
        selected.append(small_k)
    if medium_k > 0 and medium_k not in selected:
        selected.append(medium_k)

    if include_full_sample and total_langs:
        if total_langs not in selected:
            selected.append(total_langs)

    # Ensure exactly num_sizes if possible by padding geometric suggestions
    if len(selected) < num_sizes:
        geom = np.geomspace(max(1, k_min), max(k_min + 1, k_max), num=num_sizes)
        for val in geom:
            val = clamp_k(int(round(val)))
            if val not in selected:
                selected.append(val)
            if len(selected) == num_sizes:
                break

    print(f"Silhouette analysis complete. Selected sizes: {selected}")
    return sorted(selected)


def create_viz_df(tsne_df, cluster_labels, medoid_indices, data_df):
    """Attach metadata to cached t-SNE coordinates for plotting."""
    viz_df = tsne_df.copy().reset_index(drop=True)
    viz_df["cluster_id"] = cluster_labels
    viz_df["is_medoid"] = False
    viz_df.loc[medoid_indices, "is_medoid"] = True
    meta_cols = data_df[["name", "resource_category", "script", "family"]].reset_index(drop=True)
    return pd.concat([viz_df, meta_cols], axis=1)


def create_medoid_trace(medoid_df):
    hover_cols = ["resource_category", "script", "family", "cluster_id", "is_medoid", "TSNE1", "TSNE2"]
    return go.Scatter(
        x=medoid_df["TSNE1"],
        y=medoid_df["TSNE2"],
        mode="markers",
        marker=dict(size=12, color="red", symbol="circle", line=dict(width=1, color="darkred")),
        hovertext=medoid_df["name"],
        customdata=medoid_df[hover_cols],
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "TSNE1: %{customdata[5]:.3f}<br>"
            "TSNE2: %{customdata[6]:.3f}<br>"
            "Resource: %{customdata[0]}<br>"
            "Script: %{customdata[1]}<br>"
            "Family: %{customdata[2]}<br>"
            "Cluster: %{customdata[3]}<br>"
            "Medoid: %{customdata[4]}<extra></extra>"
        ),
        name="Medoids",
    )


def show_tsne_viz(tsne_df, cluster_labels, medoid_indices, data_df, method_name, output_dir, k):
    """Save Plotly scatter mirroring fineweb2_medoid_clustering.py."""
    viz_df = create_viz_df(tsne_df, cluster_labels, medoid_indices, data_df)
    hover_data = {
        "resource_category": True,
        "script": True,
        "family": True,
        "cluster_id": True,
        "is_medoid": True,
    }
    fig = px.scatter(
        viz_df,
        x="TSNE1",
        y="TSNE2",
        color="cluster_id",
        hover_name="name",
        hover_data=hover_data,
        color_continuous_scale="Viridis",
    )
    fig.add_trace(create_medoid_trace(viz_df[viz_df["is_medoid"]]))
    fig.update_layout(coloraxis_showscale=False, title=f"{method_name.upper()} clusters (k={k}) with medoids")
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{method_name}_k{k}_tsne.html")
    fig.write_html(html_path)
    print(f"Saved interactive t-SNE plot to {html_path}")


def sample_cola_families(df, embeddings_df, num_clusters, langs_per_cluster, output_dir):
    """Build CoLA-specific grouped samples from LLM embeddings."""
    print(f"Generating CoLA optimal sample: {num_clusters} clusters × {langs_per_cluster} langs")
    n_candidates = min(len(df), max(num_clusters * 3, 20))
    kmeans = KMeans(n_clusters=n_candidates, init="k-means++", n_init="auto", random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_df)
    centroids = kmeans.cluster_centers_

    centroid_dist = pairwise_distances(centroids, metric="euclidean")
    selected_cluster_ids = farthest_first_ordering(centroid_dist)[:num_clusters]

    dist_matrix = pairwise_distances(embeddings_df.values, metric="euclidean")
    selected_indices = []
    for clust_id in selected_cluster_ids:
        member_idxs = np.where(cluster_labels == clust_id)[0]
        if len(member_idxs) == 0:
            continue
        sub_dm = dist_matrix[np.ix_(member_idxs, member_idxs)]
        medoid_rel_idx = np.argmin(sub_dm.sum(axis=1))
        medoid_abs_idx = member_idxs[medoid_rel_idx]
        sorted_rel = np.argsort(sub_dm[medoid_rel_idx])
        take = min(len(member_idxs), langs_per_cluster)
        top_abs = member_idxs[sorted_rel[:take]]
        selected_indices.extend(top_abs)

    sample_df = df.iloc[selected_indices].reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    csv_name = os.path.join(output_dir, f"cola_optimal_{num_clusters}A_{langs_per_cluster}B_total{len(sample_df)}.csv")
    sample_df.to_csv(csv_name, index=False)
    print(f"Saved CoLA sample -> {csv_name}")


def cluster_and_sample(df, embeddings_df, tsne_df, sample_sizes, method_name, output_dir):
    """Cluster embeddings, order medoids, export nested subsets + visualization."""
    max_k = max(sample_sizes)
    if max_k > len(df):
        print(f"Requested k={max_k} but only {len(df)} languages. Capping…")
        max_k = len(df)
        sample_sizes = [s for s in sample_sizes if s <= max_k]

    print(f"Clustering ({method_name}) with k={max_k}…")
    kmeans = KMeans(n_clusters=max_k, init="k-means++", n_init="auto", random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_df)

    dist_matrix = pairwise_distances(embeddings_df.values, metric="euclidean")
    medoid_indices = []
    for clust_id in range(max_k):
        member_idxs = np.where(cluster_labels == clust_id)[0]
        if len(member_idxs) == 0:
            print(f"Warning: cluster {clust_id} empty.")
            continue
        sub_dm = dist_matrix[np.ix_(member_idxs, member_idxs)]
        medoid_rel_idx = np.argmin(sub_dm.sum(axis=1))
        medoid_indices.append(member_idxs[medoid_rel_idx])

    medoid_dist = dist_matrix[np.ix_(medoid_indices, medoid_indices)]
    ordered_indices = np.array(medoid_indices)[farthest_first_ordering(medoid_dist)]

    show_tsne_viz(tsne_df, cluster_labels, medoid_indices, df, method_name, output_dir, max_k)

    os.makedirs(output_dir, exist_ok=True)
    for size in sample_sizes:
        sample_df = df.iloc[ordered_indices[:size]].reset_index(drop=True)
        csv_path = os.path.join(output_dir, f"{size}_representative_mediods.csv")
        sample_df.to_csv(csv_path, index=False)
        print(f"Saved {size}-sample -> {csv_path}")


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(data_dir, "processed_artifacts")
    metadata_df, embedding_map = load_processed_artifacts(processed_dir)
    total_langs = len(metadata_df)

    # One-hot embeddings
    onehot_embeddings = embedding_map["onehot"]["embeddings"]
    onehot_tsne = embedding_map["onehot"]["tsne"]
    oh_sizes = select_optimal_sizes(
        onehot_embeddings,
        k_min=10,
        k_max=min(100, max(10, total_langs - 1)),
        include_full_sample=True,
        total_langs=total_langs,
    )
    cluster_and_sample(
        metadata_df,
        onehot_embeddings,
        onehot_tsne,
        oh_sizes,
        "onehot",
        os.path.join(data_dir, "onehot_benchmark_samples"),
    )

    # LLM embeddings
    llm_embeddings = embedding_map["llm"]["embeddings"]
    llm_tsne = embedding_map["llm"]["tsne"]
    llm_sizes = select_optimal_sizes(
        llm_embeddings,
        k_min=10,
        k_max=min(100, max(10, total_langs - 1)),
        include_full_sample=True,
        total_langs=total_langs,
    )
    cluster_and_sample(
        metadata_df,
        llm_embeddings,
        llm_tsne,
        llm_sizes,
        "llm",
        os.path.join(data_dir, "llm_benchmark_samples"),
    )

    # CoLA optimal subsets based on LLM embeddings
    cola_dir = os.path.join(data_dir, "cola_optimal_samples")
    for cfg in [(4, 4), (8, 4), (12, 4)]:
        sample_cola_families(metadata_df, llm_embeddings, cfg[0], cfg[1], cola_dir)


if __name__ == "__main__":
    main()
