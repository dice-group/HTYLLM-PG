import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

"""generate sampling plans for the tiered language budgets.

inputs:
* Tier definitions (`*_tier_language_groupings.json`)
* FineWeb language metadata (`fineweb2-language-distribution.csv`)

It emits tier-specific CSVs listing the `subset` identifiers and how many
documents to sample for each tier (12, 72, 200 languages) while respecting the
token budgets from `docs/decide_token_budget.md` under α = 0.3 smoothing.
The decisions for that are documented in `docs/decide_token_budget.md`.
"""

ALPHA = 0.3
# Tier id -> total token budget
TOKEN_BUDGETS = {
    1: 6_000_000_000,   # 12-language tier
    2: 24_000_000_000,  # 72-95 language tier
    3: 60_000_000_000,  # 200-language tier
}
TIER_FILENAMES = {
    1: "12_tier_language_groupings.json",
    2: "72_tier_language_groupings.json",
    3: "200_tier_language_groupings.json",
}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampling plans per tier.")
    parser.add_argument("--tier-dir", default=REPO_ROOT / "tools/two_stage_clustering", type=Path, help="Directory containing *_tier_language_groupings.json files.")
    parser.add_argument("--fineweb-csv", default=REPO_ROOT / "data_prep/base_data/fineweb2-language-distribution.csv", type=Path, help="FineWeb2 CSV with per-language word/document counts.")
    parser.add_argument("--output-dir", default=REPO_ROOT / "data_prep/processed_artifacts", type=Path, help="Directory where tier sampling plans will be written.")
    parser.add_argument("--alpha", default=ALPHA, type=float, help="Temperature/alpha value for smoothing the language weights.")
    return parser.parse_args()


def _load_fineweb_stats(csv_path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "train"].copy()
    for column in ("words", "documents"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    df["documents"] = df["documents"].astype(int)
    return df.set_index("subset").to_dict("index")


def _load_tier_languages(tier_dir: Path) -> Dict[int, List[str]]:
    tier_dir = tier_dir.resolve()
    tiers = {}
    for tier_id, filename in TIER_FILENAMES.items():
        path = tier_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        seen, ordered = set(), []
        for expert in data.values():
            langs = expert.get("languages") or expert.get("language") or []
            for lang in langs:
                if lang not in seen:
                    seen.add(lang)
                    ordered.append(lang)
        tiers[tier_id] = ordered
    return tiers


def _collect_language_stats(languages: Iterable[str], fineweb_stats: Dict[str, Dict[str, float]], alpha: float) -> Tuple[List[Tuple], List[str]]:
    stats: List[Tuple] = []
    missing: List[str] = []
    for subset in dict.fromkeys(languages):
        if subset not in fineweb_stats:
            missing.append(subset)
            continue

        info = fineweb_stats[subset]
        words = float(info.get("words", 0.0))
        documents = int(info.get("documents", 0))
        if words <= 0 or documents <= 0:
            # Cannot sample languages with no words or documents recorded.
            missing.append(subset)
            continue

        tokens_per_doc = words / documents
        weight = words ** alpha if words > 0 else 0.0
        stats.append((subset, words, documents, tokens_per_doc, weight))
    return stats, missing


def _build_plan_for_tier(languages: Iterable[str], fineweb_stats: Dict[str, Dict[str, float]], tier_id: int, alpha: float, token_budget: int) -> Tuple[pd.DataFrame, Dict]:
    lang_stats, missing = _collect_language_stats(languages, fineweb_stats, alpha)
    if not lang_stats:
        raise ValueError(f"No valid languages for tier {tier_id}. Missing: {missing}")

    total_weight = sum(stat[4] for stat in lang_stats)
    allocations = []
    total_effective_tokens = 0.0
    for subset, words, documents, tokens_per_doc, weight in lang_stats:
        share = weight / total_weight if total_weight > 0 else 0.0
        target_tokens = share * token_budget
        capped_tokens = min(target_tokens, words)
        docs_target = int(capped_tokens / tokens_per_doc)
        if capped_tokens > 0 and docs_target == 0:
            docs_target = 1
        docs_target = min(docs_target, documents)
        effective_tokens = docs_target * tokens_per_doc
        total_effective_tokens += effective_tokens

        allocations.append(
            {
                "subset": subset,
                "documents": docs_target,
                "tokens_budgeted": round(target_tokens),
                "tokens_effective": round(effective_tokens),
                "words_available": round(words),
                "documents_available": documents,
                "tokens_per_document": tokens_per_doc,
            }
        )

    allocations_df = pd.DataFrame(sorted(allocations, key=lambda row: row["subset"]))
    summary = {
        "tier": tier_id,
        "languages": len(lang_stats),
        "token_budget": token_budget,
        "alpha": alpha,
        "tokens_requested": round(total_effective_tokens),
        "tokens_shortfall": max(token_budget - total_effective_tokens, 0),
        "missing_languages": missing,
    }
    return allocations_df, summary


def main() -> None:
    args = parse_args()
    fineweb_stats = _load_fineweb_stats(args.fineweb_csv)
    tiers = _load_tier_languages(args.tier_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for tier_id, languages in tiers.items():
        if tier_id not in TOKEN_BUDGETS or not languages:
            continue
        plan_df, summary = _build_plan_for_tier(
            languages,
            fineweb_stats,
            tier_id,
            args.alpha,
            TOKEN_BUDGETS[tier_id],
        )
        tier_size = summary["languages"]
        csv_name = f"sampling_plan_tier{tier_id}_{tier_size}langs.csv"
        plan_path = output_dir / csv_name
        plan_df.to_csv(plan_path, index=False)
        summary["plan_path"] = str(plan_path)
        summaries.append(summary)

    summary_path = output_dir / "sampling_plan_summary.json"
    summary_path.write_text(json.dumps({"tiers": summaries}, indent=2))


if __name__ == "__main__":
    main()
