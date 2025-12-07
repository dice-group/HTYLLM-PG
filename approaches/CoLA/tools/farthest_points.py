import argparse
import pandas as pd
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
from sklearn.manifold import MDS
from umap import UMAP


DEFAULT_OUTPUT = Path("data_prep/processed_artifacts/farthest_points.png")
DEFAULT_METADATA_CSV = Path("data_prep/base_data/fineweb2-language-distribution.csv")


@dataclass
class SelectionOutcome:
    k: int
    indices: List[int]
    neighbor_map: Dict[int, List[int]]
    metrics: Dict[str, object]
    score: float


def load_distance_data(npz_path: Path) -> tuple[np.ndarray, List[str]]:
    """Load a distance matrix and language codes from an NPZ archive."""
    data = np.load(npz_path)
    matrix = data["matrix"].astype(float)
    codes = [str(code) for code in data["codes"]]
    return matrix, codes


def load_coordinates(csv_path: Path, codes: Sequence[str]) -> np.ndarray:
    """Load 2-D coordinates for each code."""
    mapping: Dict[str, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"code", "x", "y"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Coordinate CSV must contain {required}.")
        for row in reader:
            code = row["code"]
            mapping[code] = (float(row["x"]), float(row["y"]))
    coords = []
    missing = []
    for code in codes:
        if code not in mapping:
            missing.append(code)
        else:
            coords.append(mapping[code])
    if missing:
        raise ValueError(f"Missing coordinates for: {missing[:5]}")
    return np.asarray(coords, dtype=float)


def select_farthest_points(dist_matrix: np.ndarray, k: int, start: int | None = None) -> List[int]:
    """Return indices via greedy farthest-point sampling over the distance matrix."""
    D = np.asarray(dist_matrix)
    n = D.shape[0]

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if k <= 0 or n == 0:
        return []
    if np.any(D < 0):
        raise ValueError("Distance matrix cannot contain negative values.")
    if np.isnan(D).any():
        raise ValueError("Distance matrix cannot contain NaNs.")

    k = min(k, n)

    if start is None:
        start = int(np.argmax(D.sum(axis=1)))
    else:
        start = int(np.clip(start, 0, n - 1))

    selected = [start]
    min_d = D[start].astype(float).copy()
    min_d[start] = -np.inf

    for _ in range(1, k):
        candidate = int(np.argmax(min_d))
        selected.append(candidate)
        min_d = np.minimum(min_d, D[candidate])
        min_d[candidate] = -np.inf

    return selected


def _neighbors_for_count(dist_matrix: np.ndarray, idx: int, count: int) -> List[int]:
    if count <= 0:
        return []
    row = dist_matrix[idx].astype(float).copy()
    row[idx] = np.inf
    order = np.argsort(row)
    chosen = []
    for candidate in order:
        if not np.isfinite(row[candidate]):
            continue
        chosen.append(int(candidate))
        if len(chosen) == count:
            break
    return chosen


def compute_neighbors(dist_matrix: np.ndarray, selected: Sequence[int], counts: Sequence[int]) -> Dict[int, List[int]]:
    """Return the closest neighbors for each selected index."""
    if len(selected) != len(counts):
        raise ValueError("Counts must match the selected indices.")
    neighbors = {}
    for idx, count in zip(selected, counts):
        neighbors[int(idx)] = _neighbors_for_count(dist_matrix, int(idx), int(count))
    return neighbors


def allocate_neighbor_counts(k: int, target_total: int | None, neighbors_per_language: int | None) -> List[int]:
    if neighbors_per_language is not None:
        return [max(0, neighbors_per_language)] * k
    remaining = max(0, (target_total or k) - k)
    if k == 0:
        return []
    base = remaining // k
    extras = remaining % k
    return [base + (1 if idx < extras else 0) for idx in range(k)]


def _pairwise_stats(submatrix: np.ndarray) -> Dict[str, float | None]:
    if submatrix.shape[0] < 2:
        return {"min": None, "mean": None, "median": None}
    mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
    values = submatrix[mask]
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def _neighbor_stats(dist_matrix: np.ndarray, neighbor_map: Dict[int, List[int]]) -> Dict[str, float | None]:
    dists = []
    per_language = {}
    for center, neighbors in neighbor_map.items():
        if not neighbors:
            per_language[center] = None
            continue
        values = dist_matrix[center, neighbors]
        per_language[center] = float(values.mean())
        dists.extend(values.tolist())
    if not dists:
        summary = {"mean": None, "median": None, "max": None}
    else:
        arr = np.asarray(dists, dtype=float)
        summary = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }
    return {"summary": summary, "per_language": per_language}


def compute_quality_metrics(dist_matrix: np.ndarray, selected: Sequence[int], neighbor_map: Dict[int, List[int]]) -> Dict[str, object]:
    sub = dist_matrix[np.ix_(selected, selected)]
    pairwise = _pairwise_stats(sub)
    neighbor_stats = _neighbor_stats(dist_matrix, neighbor_map)
    return {
        "farthest_pairwise": pairwise,
        "neighbor_coherence": neighbor_stats,
    }


def compute_quality_score(metrics: Dict[str, object]) -> float:
    spread = metrics["farthest_pairwise"]["min"]
    neighbor_mean = metrics["neighbor_coherence"]["summary"]["mean"]
    if spread is None:
        return float("-inf")
    if neighbor_mean is None:
        neighbor_mean = 0.0
    return float(spread) - float(neighbor_mean)


def embed_points(dist_matrix: np.ndarray) -> np.ndarray:
    """2-D MDS embedding of a precomputed distance matrix."""
    if dist_matrix.shape[0] < 2:
        return np.zeros((dist_matrix.shape[0], 2))
    model = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        normalized_stress="auto",
    )
    return model.fit_transform(dist_matrix)


def embed_points_umap(dist_matrix: np.ndarray) -> np.ndarray:
    """2-D UMAP embedding from a precomputed distance matrix."""
    if dist_matrix.shape[0] < 2:
        return np.zeros((dist_matrix.shape[0], 2))
    model = UMAP(
        n_components=2,
        metric="precomputed",
        random_state=0,
        init="spectral",
        n_neighbors=min(15, dist_matrix.shape[0] - 1),
    )
    return model.fit_transform(dist_matrix)


def plot_points(
    coords: np.ndarray,
    labels: Sequence[str],
    assignments: Dict[int, List[int]],
    output_path: Path,
    show_background: bool,
) -> None:
    """Save a scatter plot highlighting selected indices."""
    coords = np.asarray(coords)
    fig, ax = plt.subplots(figsize=(6, 6))
    if show_background and len(coords):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color="#bbbbbb",
            s=15,
            alpha=0.35,
            label="All languages",
        )

    color_cycle = plt.colormaps["tab10"]
    selected = list(assignments.keys())
    for idx, sel in enumerate(selected):
        color = color_cycle(idx % 10)
        neighbor_ids = assignments[sel]
        if neighbor_ids:
            neighbor_coords = coords[neighbor_ids]
            neighbor_labels = [labels[i] for i in neighbor_ids]
            ax.scatter(
                neighbor_coords[:, 0],
                neighbor_coords[:, 1],
                color=color,
                s=25,
                alpha=0.8,
            )
            for (x, y), label in zip(neighbor_coords, neighbor_labels):
                ax.text(x, y, label, fontsize=7, ha="center", va="center")

        ax.scatter(
            coords[sel, 0],
            coords[sel, 1],
            color=color,
            s=70,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.text(
            coords[sel, 0],
            coords[sel, 1],
            labels[sel],
            fontsize=9,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_title("Farthest languages (precomputed distances)")
    if selected:
        handles = []
        if show_background:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#bbbbbb",
                    markersize=6,
                    label="All languages",
                )
            )
        handles.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_cycle(0),
                    markersize=6,
                    label="Neighbors",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markerfacecolor=color_cycle(1),
                    markersize=8,
                    label="Farthest",
                ),
            ]
        )
        ax.legend(handles=handles, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and visualize farthest languages."
    )
    add = parser.add_argument
    add("--distance-npz", type=Path, required=True)
    add("--k", type=int, help="Override K sweep.")
    add("--k-min", type=int, default=4)
    add("--k-max", type=int, default=16)
    add("--target-total", type=int, help="Final language count (seed+neighbors).")
    add("--neighbors-per-language", type=int, help="(Legacy) fixed neighbor count per seed.")
    add("--output-image", type=Path, default=Path("data_prep/processed_artifacts/farthest_points.png"))
    add("--output-umap-image", type=Path)
    add("--json-output", type=Path)
    add("--coordinate-csv", type=Path)
    add("--limit-languages", type=int)
    add("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    add("--min-documents", type=int)
    add("--show-all", action="store_true")
    args = parser.parse_args()

    matrix, codes = load_distance_data(args.distance_npz)
    matrix, codes = filter_languages(
        matrix,
        codes,
        args.metadata_csv,
        args.min_documents,
    )
    if args.limit_languages is not None:
        limit = max(1, min(args.limit_languages, matrix.shape[0]))
        matrix = matrix[:limit, :limit]
        codes = codes[:limit]

    if args.target_total is None and args.neighbors_per_language is None:
        raise ValueError("Specify either --target-total or --neighbors-per-language.")
    if args.target_total is not None and args.target_total <= 0:
        raise ValueError("--target-total must be positive.")
    use_neighbors_per_lang = False
    if args.target_total is not None:
        neighbors_per_target = None
    else:
        neighbors_per_target = args.neighbors_per_language
        use_neighbors_per_lang = True

    coord_override = None
    if args.coordinate_csv is not None:
        coord_override = load_coordinates(args.coordinate_csv, codes)

    if args.k is not None:
        k_values = [args.k]
    else:
        k_values = list(
            range(
                max(1, args.k_min),
                max(1, args.k_max) + 1,
            )
        )

    def evaluate_k(k: int) -> SelectionOutcome | None:
        if k <= 0 or k > matrix.shape[0]:
            return None
        indices = select_farthest_points(matrix, k)
        counts = allocate_neighbor_counts(
            len(indices),
            None if use_neighbors_per_lang else args.target_total,
            neighbors_per_target if use_neighbors_per_lang else None,
        )
        neighbor_map = compute_neighbors(matrix, indices, counts)
        metrics = compute_quality_metrics(matrix, indices, neighbor_map)
        return SelectionOutcome(
            k=k,
            indices=indices,
            neighbor_map=neighbor_map,
            metrics=metrics,
            score=compute_quality_score(metrics),
        )

    evaluated = []
    for k in k_values:
        outcome = evaluate_k(k)
        if outcome is not None:
            evaluated.append(outcome)

    if not evaluated:
        raise ValueError("No valid K values to evaluate.")

    best = max(evaluated, key=lambda item: item.score)
    indices = best.indices
    neighbor_map = best.neighbor_map
    quality_metrics = best.metrics
    selected_codes = [codes[i] for i in indices]
    neighbor_indices = sorted(
        {idx for lst in neighbor_map.values() for idx in lst}
    )

    if args.show_all:
        coords = coord_override if coord_override is not None else embed_points(matrix)
        plot_points(
            coords,
            codes,
            {idx: neighbor_map[idx] for idx in indices},
            args.output_image,
            show_background=True,
        )
        if args.output_umap_image is not None and coord_override is None:
            coords_umap = embed_points_umap(matrix)
            plot_points(
                coords_umap,
                codes,
                {idx: neighbor_map[idx] for idx in indices},
                args.output_umap_image,
                show_background=True,
            )
    else:
        subset_extra = [
            idx for idx in neighbor_indices if idx not in indices
        ]
        subset_order = indices + subset_extra
        subset_matrix = matrix[np.ix_(subset_order, subset_order)]
        subset_codes = [codes[i] for i in subset_order]

        index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(subset_order)}
        remapped = {
            index_map[sel]: [index_map[n] for n in neighbor_map[sel] if n in index_map]
            for sel in indices
        }

        coords = (
            coord_override[subset_order]
            if coord_override is not None
            else embed_points(subset_matrix)
        )
        plot_points(
            coords,
            subset_codes,
            remapped,
            args.output_image,
            show_background=False,
        )
        if args.output_umap_image is not None and coord_override is None:
            coords_umap = embed_points_umap(subset_matrix)
            plot_points(
                coords_umap,
                subset_codes,
                remapped,
                args.output_umap_image,
                show_background=False,
            )

    print(f"Best K={best.k} (score={best.score:.4f})")
    print("Selected languages:", ", ".join(selected_codes))
    spread = quality_metrics["farthest_pairwise"]
    if spread["min"] is not None:
        print(
            f"Pairwise spread - min: {spread['min']:.4f}, "
            f"median: {spread['median']:.4f}, mean: {spread['mean']:.4f}"
        )
    neighbor_summary = quality_metrics["neighbor_coherence"]["summary"]
    if neighbor_summary["mean"] is not None:
        print(
            f"Neighbor distance - mean: {neighbor_summary['mean']:.4f}, "
            f"median: {neighbor_summary['median']:.4f}, "
            f"max: {neighbor_summary['max']:.4f}"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        per_k = [
            {
                "k": entry.k,
                "score": entry.score,
                "codes": [codes[i] for i in entry.indices],
                "quality_metrics": entry.metrics,
            }
            for entry in evaluated
        ]
        best_neighbors = {
            codes[idx]: [codes[n] for n in neighbors]
            for idx, neighbors in neighbor_map.items()
        }
        serialized = {
            "best_k": best.k,
            "best_score": best.score,
            "best_codes": selected_codes,
            "best_neighbors": best_neighbors,
            "quality_metrics": {
                "farthest_pairwise": spread,
                "neighbor_coherence": {
                    "summary": neighbor_summary,
                    "per_language": {
                        codes[idx]: value
                        for idx, value in quality_metrics[
                            "neighbor_coherence"
                        ]["per_language"].items()
                    },
                },
            },
            "per_k": per_k,
        }
        args.json_output.write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )


def filter_languages(
    matrix: np.ndarray,
    codes: List[str],
    metadata_csv: Path | None,
    min_documents: int | None,
) -> tuple[np.ndarray, List[str]]:
    if metadata_csv is None or min_documents is None:
        return matrix, codes

    df = pd.read_csv(metadata_csv)

    aggregated = df.groupby("code")["documents"].max().reset_index()
    valid_codes = set(
        aggregated.loc[
            aggregated["documents"] >= min_documents, "code"
        ].tolist()
    )
    mask = [code in valid_codes for code in codes]
    indices = [i for i, keep_flag in enumerate(mask) if keep_flag]
    filtered_matrix = matrix[np.ix_(indices, indices)]
    filtered_codes = [codes[i] for i in indices]
    return filtered_matrix, filtered_codes


if __name__ == "__main__":
    main()