import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def _format_top(counter: Counter, limit: int = 5) -> str:
    if not counter:
        return ""
    items = []
    for name, count in counter.most_common(limit):
        items.append(f"{name}({count})")
    return "; ".join(items)


def _load_groupings(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _language_family(metadata: dict, lang: str) -> str:
    entry = metadata.get(lang, {})
    family = entry.get("family")
    return str(family) if family else "Unknown"


def _language_name(metadata: dict, lang: str) -> str:
    entry = metadata.get(lang, {})
    name = entry.get("name")
    return str(name) if name else lang


def _language_script(lang: str) -> str:
    if "_" in lang:
        return lang.split("_", 1)[1]
    return "Unknown"


def _summarize_languages(
    languages: Iterable[str],
    metadata: dict,
    top_limit: int = 5,
) -> Tuple[str, str, str]:
    lang_list = list(languages)
    family_counts = Counter(_language_family(metadata, lang) for lang in lang_list)
    script_counts = Counter(_language_script(lang) for lang in lang_list)
    name_counts = Counter(_language_name(metadata, lang) for lang in lang_list)
    return (
        _format_top(family_counts, limit=top_limit),
        _format_top(script_counts, limit=top_limit),
        _format_top(name_counts, limit=top_limit),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize expert/subexpert language composition.")
    parser.add_argument("--language-map", default="tools/two_stage_clustering/200_tier_language_groupings.json")
    parser.add_argument("--expert-delta", default="result_analysis/paper_eval_summary/expert_delta_vs_colaflat.csv")
    parser.add_argument("--output-dir", default="result_analysis/paper_eval_summary")
    parser.add_argument("--top-limit", type=int, default=5)
    args = parser.parse_args()

    groupings = _load_groupings(Path(args.language_map))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_rows = []
    subexpert_rows = []
    for expert_id, entry in groupings.items():
        languages = entry.get("languages") or []
        metadata = entry.get("metadata") or {}
        top_families, top_scripts, top_languages = _summarize_languages(
            languages, metadata, top_limit=args.top_limit
        )
        subgroups = entry.get("subgroups") or {}
        expert_rows.append(
            {
                "expert_id": str(expert_id),
                "language_count": len(languages),
                "subexpert_count": len(subgroups) if isinstance(subgroups, dict) else 0,
                "top_families": top_families,
                "top_scripts": top_scripts,
                "top_languages": top_languages,
            }
        )
        if isinstance(subgroups, dict):
            for sub_id, sub_langs in subgroups.items():
                sub_families, sub_scripts, sub_langs_fmt = _summarize_languages(
                    sub_langs, metadata, top_limit=args.top_limit
                )
                subexpert_rows.append(
                    {
                        "expert_id": str(expert_id),
                        "subexpert_id": str(sub_id),
                        "language_count": len(sub_langs),
                        "top_families": sub_families,
                        "top_scripts": sub_scripts,
                        "top_languages": sub_langs_fmt,
                    }
                )

    expert_df = pd.DataFrame(expert_rows)
    subexpert_df = pd.DataFrame(subexpert_rows)
    expert_path = output_dir / "expert_language_summary.csv"
    subexpert_path = output_dir / "subexpert_language_summary.csv"
    expert_df.to_csv(expert_path, index=False)
    subexpert_df.to_csv(subexpert_path, index=False)

    delta_path = Path(args.expert_delta)
    if delta_path.exists():
        delta_df = pd.read_csv(delta_path)
        if "expert_id" in delta_df.columns:
            delta_df["expert_id"] = delta_df["expert_id"].astype(str)
        merged = expert_df.merge(delta_df, on="expert_id", how="left")
        merged_path = output_dir / "expert_delta_family_summary.csv"
        merged.to_csv(merged_path, index=False)

    print(f"Wrote {expert_path}")
    print(f"Wrote {subexpert_path}")
    if delta_path.exists():
        print(f"Wrote {output_dir / 'expert_delta_family_summary.csv'}")


if __name__ == "__main__":
    main()
