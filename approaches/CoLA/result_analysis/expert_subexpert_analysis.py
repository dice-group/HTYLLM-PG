import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _load_language_maps(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expert_map: Dict[str, str] = {}
    subexpert_map: Dict[str, str] = {}
    for expert_id, entry in data.items():
        languages = entry.get("languages") or []
        for lang in languages:
            expert_map[lang] = str(expert_id)
        subgroups = entry.get("subgroups") or {}
        if isinstance(subgroups, dict):
            for sub_id, sub_langs in subgroups.items():
                for lang in sub_langs:
                    subexpert_map[lang] = str(sub_id)
    return expert_map, subexpert_map


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols)
        .agg(
            language_count=("language", "nunique"),
            task_count=("task", "count"),
            acc_mean=("acc_norm", "mean"),
            acc_median=("acc_norm", "median"),
            acc_std=("acc_norm", "std"),
            acc_min=("acc_norm", "min"),
            acc_max=("acc_norm", "max"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-language scores by expert/subexpert.")
    parser = argparse.ArgumentParser(description="Aggregate per-language scores by expert/subexpert.")
    parser.add_argument("--per-task-scores", default="result_analysis/paper_eval_summary/per_task_scores.csv")
    parser.add_argument("--language-map", default="tools/two_stage_clustering/200_tier_language_groupings.json")
    parser.add_argument("--output-dir", default="result_analysis/paper_eval_summary")
    parser.add_argument("--drop-unmapped", action="store_true", help="Drop languages that do not appear in the clustering map.")
    args = parser.parse_args()

    per_task_path = Path(args.per_task_scores)
    language_map_path = Path(args.language_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_task_path)
    expert_map, subexpert_map = _load_language_maps(language_map_path)

    df["expert_id"] = df["language"].map(expert_map).fillna("unmapped")
    df["subexpert_id"] = df["language"].map(subexpert_map).fillna("unmapped")
    if args.drop_unmapped:
        df = df[df["expert_id"] != "unmapped"].copy()

    expert_summary = _summarize(df, ["label", "expert_id"])
    subexpert_summary = _summarize(df, ["label", "expert_id", "subexpert_id"])

    expert_path = output_dir / "expert_summary.csv"
    subexpert_path = output_dir / "subexpert_summary.csv"
    expert_summary.to_csv(expert_path, index=False)
    subexpert_summary.to_csv(subexpert_path, index=False)

    unmapped = df[df["expert_id"] == "unmapped"]["language"].drop_duplicates().sort_values()
    if not unmapped.empty:
        unmapped_path = output_dir / "unmapped_languages.txt"
        unmapped_path.write_text("\n".join(unmapped.tolist()), encoding="utf-8")

    print(f"Wrote {expert_path}")
    print(f"Wrote {subexpert_path}")


if __name__ == "__main__":
    main()
