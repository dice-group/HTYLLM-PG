import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VARIANT_ORDER = [
    "colaflat",
    "colaexp-headbias",
    "colaexp-hard",
    "colaexp-lpr",
    "hydra-exp-lpr",
]


def _pivot_acc(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=index_cols,
        columns="label",
        values="acc_mean",
        aggfunc="mean",
    )
    return pivot.reindex(columns=[c for c in VARIANT_ORDER if c in pivot.columns])


def _add_counts(base: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    joined = base.join(counts)
    joined.reset_index(inplace=True)
    return joined


def _save_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    center: float | None = None,
    cbar_label: str | None = None,
) -> None:
    mask = np.isnan(data)
    if center is not None:
        max_abs = np.nanmax(np.abs(data))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = None, None

    fig_w = max(6.5, len(col_labels) * 1.3)
    fig_h = max(4.2, len(row_labels) * 0.40)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f0f0f0")

    im = ax.imshow(
        np.ma.masked_array(data, mask),
        aspect="auto",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)

    # Subtle grid to separate cells.
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_tables(df: pd.DataFrame, index_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = _pivot_acc(df, index_cols)
    counts = df.groupby(index_cols)["language_count"].max().to_frame("language_count")
    base = _add_counts(pivot, counts)

    delta = pivot.copy()
    if "colaflat" in delta.columns:
        for col in delta.columns:
            if col == "colaflat":
                continue
            delta[col] = delta[col] - delta["colaflat"]
    delta = _add_counts(delta, counts)
    return base, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Create expert/subexpert tables and heatmaps.")
    parser.add_argument("--expert-summary", default="result_analysis/paper_eval_summary/expert_summary.csv")
    parser.add_argument("--subexpert-summary", default="result_analysis/paper_eval_summary/subexpert_summary.csv")
    parser.add_argument("--output-dir", default="result_analysis/paper_eval_summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_df = pd.read_csv(args.expert_summary)
    subexpert_df = pd.read_csv(args.subexpert_summary)

    expert_table, expert_delta = _build_tables(expert_df, ["expert_id"])
    expert_table.to_csv(output_dir / "expert_table.csv", index=False)
    expert_delta.to_csv(output_dir / "expert_delta_vs_colaflat.csv", index=False)

    subexpert_df = subexpert_df.copy()
    subexpert_df["subexpert_key"] = subexpert_df["expert_id"].astype(str) + ":" + subexpert_df["subexpert_id"].astype(str)
    subexpert_table, subexpert_delta = _build_tables(subexpert_df, ["subexpert_key"])
    subexpert_table.to_csv(output_dir / "subexpert_table.csv", index=False)
    subexpert_delta.to_csv(output_dir / "subexpert_delta_vs_colaflat.csv", index=False)

    expert_pivot = _pivot_acc(expert_df, ["expert_id"]).sort_index()
    expert_counts = expert_df.groupby("expert_id")["language_count"].max()
    expert_labels = [
        f"E{idx} (n={int(expert_counts.get(idx, 0))})" for idx in expert_pivot.index
    ]
    _save_heatmap(
        expert_pivot.to_numpy(),
        expert_labels,
        expert_pivot.columns.tolist(),
        "Expert accuracy (macro acc_norm)",
        output_dir / "expert_accuracy_heatmap.png",
        cmap="viridis",
        cbar_label="acc_norm (macro)",
    )

    expert_delta_pivot = expert_pivot.copy()
    if "colaflat" in expert_delta_pivot.columns:
        for col in expert_delta_pivot.columns:
            if col == "colaflat":
                expert_delta_pivot[col] = 0.0
            else:
                expert_delta_pivot[col] = expert_delta_pivot[col] - expert_delta_pivot["colaflat"]
    _save_heatmap(
        expert_delta_pivot.to_numpy(),
        expert_labels,
        expert_delta_pivot.columns.tolist(),
        "Expert delta vs colaflat",
        output_dir / "expert_delta_heatmap.png",
        cmap="coolwarm",
        center=0.0,
        cbar_label="Δ acc_norm vs CoLA flat",
    )

    subexpert_pivot = _pivot_acc(subexpert_df, ["subexpert_key"]).sort_index()
    subexpert_counts = subexpert_df.groupby("subexpert_key")["language_count"].max()
    subexpert_labels = [
        f"{idx} (n={int(subexpert_counts.get(idx, 0))})" for idx in subexpert_pivot.index
    ]
    _save_heatmap(
        subexpert_pivot.to_numpy(),
        subexpert_labels,
        subexpert_pivot.columns.tolist(),
        "Subexpert accuracy (macro acc_norm)",
        output_dir / "subexpert_accuracy_heatmap.png",
        cmap="viridis",
        cbar_label="acc_norm (macro)",
    )

    subexpert_delta_pivot = subexpert_pivot.copy()
    if "colaflat" in subexpert_delta_pivot.columns:
        for col in subexpert_delta_pivot.columns:
            if col == "colaflat":
                subexpert_delta_pivot[col] = 0.0
            else:
                subexpert_delta_pivot[col] = subexpert_delta_pivot[col] - subexpert_delta_pivot["colaflat"]
    _save_heatmap(
        subexpert_delta_pivot.to_numpy(),
        subexpert_labels,
        subexpert_delta_pivot.columns.tolist(),
        "Subexpert delta vs colaflat",
        output_dir / "subexpert_delta_heatmap.png",
        cmap="coolwarm",
        center=0.0,
        cbar_label="Δ acc_norm vs CoLA flat",
    )

    print(f"Wrote tables and heatmaps to {output_dir}")


if __name__ == "__main__":
    main()
