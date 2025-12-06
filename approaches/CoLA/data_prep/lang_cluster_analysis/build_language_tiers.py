import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FW_LANG_PATH = ROOT / "data_prep" / "base_data" / "fineweb2-language-distribution.csv"
LANG2VEC_DIR = ROOT / "data_prep" / "lang_cluster_analysis" / "lang2vec"
sys.path.insert(0, str(LANG2VEC_DIR))
from lang2vec import lang2vec as l2v

RESOURCE_PRIORITY = ["low", "medium", "high"]
DEFAULT_TIERS = [
    {"name": "tier12", "min_total": 12, "max_total": 12},
    {"name": "tier95", "min_total": 72, "max_total": 95},
    {"name": "tier200", "min_total": 195, "max_total": 205},
    {"name": "tier635", "min_total": 620, "max_total": 650},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build nested language tiers using Lang2Vec distances.")
    parser.add_argument("--distance", action="append", choices=l2v.DISTANCES, help="Distance types to include; can be repeated.")
    parser.add_argument("--distance-weight", action="append", default=[], help="Optional distance:weight pairs.")
    parser.add_argument("--tier", action="append", help="Override tier as name:min:max.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Inter-family distance weight.")
    parser.add_argument("--beta", type=float, default=1.0, help="Intra-family distance weight.")
    parser.add_argument("--block-size", type=int, default=2, help="Languages emitted per family per cycle.")
    parser.add_argument("--output", default=ROOT / "docs" / "language_tiers.json", help="Output file path.")
    return parser.parse_args()


def load_data():
    cols = ["code", "family", "resource_category"]
    try:
        df = pd.read_csv(FW_LANG_PATH, usecols=cols).drop_duplicates("code")
    except ValueError:
        df = pd.read_csv(FW_LANG_PATH).drop_duplicates("code")
        if "resource_category" not in df.columns:
            df["resource_category"] = "unknown"
        df = df[cols]
    distance_langs = set(getattr(l2v, "DISTANCE_LANGUAGES", l2v.available_distance_languages()))
    df = df[df["code"].isin(distance_langs)].reset_index(drop=True)
    # normalize resource labels
    df["resource_category"] = df["resource_category"].fillna("unknown").str.lower()
    return df


def build_distance_matrix(codes: List[str], distances: List[str], weights: Dict[str, float]) -> np.ndarray:
    norm = sum(weights.values())
    if norm <= 0:
        raise ValueError("Distance weights must sum to a positive value.")
    matrix = np.zeros((len(codes), len(codes)), dtype=float)
    for dist in distances:
        weight = weights[dist] / norm
        chunk = np.asarray(l2v.distance(dist, codes), dtype=float)
        matrix += weight * chunk
    np.fill_diagonal(matrix, 0.0)
    return matrix


def compute_medoid(matrix: np.ndarray, indices: List[int]) -> int:
    if len(indices) == 1:
        return indices[0]
    sub = matrix[np.ix_(indices, indices)]
    local = int(np.argmin(sub.mean(axis=1)))
    return indices[local]


def order_families(matrix: np.ndarray, families: pd.Series) -> Tuple[List[str], Dict[str, List[int]]]:
    fam_names = sorted(families.unique())
    family_indices = {fam: np.where(families == fam)[0].tolist() for fam in fam_names}
    medoid_idx = [compute_medoid(matrix, family_indices[fam]) for fam in fam_names]
    medoid_dist = matrix[np.ix_(medoid_idx, medoid_idx)]
    order = []
    remaining = fam_names.copy()
    first = max(remaining, key=lambda fam: medoid_dist[fam_names.index(fam)].mean())
    order.append(first)
    remaining.remove(first)
    while remaining:
        best = max(
            remaining,
            key=lambda fam: min(
                medoid_dist[fam_names.index(fam), fam_names.index(sel)] for sel in order
            ),
        )
        order.append(best)
        remaining.remove(best)
    return order, family_indices


def resource_priority_list(available: List[str]) -> List[str]:
    present = [cat for cat in RESOURCE_PRIORITY if cat in available]
    remainder = [cat for cat in available if cat not in present]
    return present + remainder


def order_languages_for_family(matrix: np.ndarray, family_indices: List[int], resource_labels: List[str]) -> List[int]:
    sub = matrix[np.ix_(family_indices, family_indices)]
    local_indices = list(range(len(family_indices)))
    if len(local_indices) == 1:
        return [family_indices[0]]
    medoid_local = int(np.argmin(sub.mean(axis=1)))
    selected = [medoid_local]
    remaining = set(local_indices)
    remaining.remove(medoid_local)
    covered_categories = {resource_labels[medoid_local]}
    cycle = resource_priority_list([resource_labels[idx] for idx in local_indices])

    def pick_candidate(candidates: List[int]) -> int:
        if not selected:
            return candidates[0]
        best = None
        best_score = None
        for candidate in candidates:
            dists = sub[candidate, selected]
            score = float(np.mean(dists))
            if best is None or score < best_score:
                best = candidate
                best_score = score
        return best

    while remaining:
        target = next((cat for cat in cycle if cat not in covered_categories and cat in {resource_labels[idx] for idx in remaining}), None)
        if target:
            candidates = [idx for idx in remaining if resource_labels[idx] == target]
        else:
            candidates = list(remaining)
        chosen = pick_candidate(candidates)
        selected.append(chosen)
        remaining.remove(chosen)
        covered_categories.add(resource_labels[chosen])
    return [family_indices[idx] for idx in selected]


def build_family_language_orders(matrix: np.ndarray, df: pd.DataFrame, family_order: List[str],family_indices: Dict[str, List[int]]) -> Dict[str, List[int]]:
    result = {}
    for fam in family_order:
        idxs = family_indices[fam]
        labels = df.iloc[idxs]["resource_category"].tolist()
        result[fam] = order_languages_for_family(matrix, idxs, labels)
    return result


def compute_inter_intra(matrix: np.ndarray, selected_idx: List[int], families: List[str]) -> Tuple[float, float]:
    if len(selected_idx) < 2:
        return 0.0, 0.0
    sub = matrix[np.ix_(selected_idx, selected_idx)]
    fam_arr = np.array(families)
    inter_vals = []
    intra_vals = []
    for i in range(len(selected_idx)):
        for j in range(i + 1, len(selected_idx)):
            if fam_arr[i] == fam_arr[j]:
                intra_vals.append(sub[i, j])
            else:
                inter_vals.append(sub[i, j])
    inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    intra = float(np.mean(intra_vals)) if intra_vals else 0.0
    return inter, intra


def build_language_sequence(family_order: List[str], family_lang_order: Dict[str, List[int]], block_size: int) -> List[Tuple[str, int]]:
    sequence: List[Tuple[str, int]] = []
    if block_size <= 0:
        block_size = 1
    max_len = max(len(vals) for vals in family_lang_order.values())
    for depth in range(0, max_len, block_size):
        for fam in family_order:
            langs = family_lang_order[fam]
            for offset in range(block_size):
                idx = depth + offset
                if idx < len(langs):
                    sequence.append((fam, langs[idx]))
    return sequence


def search_tier_by_prefix(tier_cfg: Dict[str, int], sequence: List[Tuple[str, int]], df: pd.DataFrame, matrix: np.ndarray, alpha: float, beta: float, prev_total: int) -> Dict:
    start_total = max(prev_total, tier_cfg["min_total"])
    end_total = min(len(sequence), tier_cfg["max_total"])
    if start_total > end_total:
        raise RuntimeError(f"Tier {tier_cfg['name']} cannot be satisfied within available languages")
    best = None
    for total in range(start_total, end_total + 1):
        subset = sequence[:total]
        idxs = [item[1] for item in subset]
        families = [item[0] for item in subset]
        inter, intra = compute_inter_intra(matrix, idxs, families)
        score = alpha * inter - beta * intra
        if best is None or score > best["score"]:
            per_family: Dict[str, List[str]] = {}
            for fam, idx in subset:
                per_family.setdefault(fam, []).append(df.iloc[idx]["code"])
            best = {
                "total": total,
                "score": score,
                "inter": inter,
                "intra": intra,
                "codes": [df.iloc[idx]["code"] for idx in idxs],
                "per_family": per_family,
            }
    if not best:
        raise RuntimeError(f"No valid prefix found for tier {tier_cfg['name']}")
    return best


def main():
    args = parse_args()
    distances = args.distance or ["genetic"]
    if args.distance_weight:
        weights = {}
        for spec in args.distance_weight:
            name, val = spec.split(":", 1)
            if name not in distances:
                raise ValueError(f"Weight specified for unknown distance {name}")
            weights[name] = float(val)
        for d in distances:
            weights.setdefault(d, 1.0)
    else:
        weights = {d: 1.0 for d in distances}

    if args.tier:
        tiers = []
        for spec in args.tier:
            name, min_val, max_val = spec.split(":", 2)
            tiers.append({"name": name, "min_total": int(min_val), "max_total": int(max_val)})
    else:
        tiers = DEFAULT_TIERS

    df = load_data()
    codes = df["code"].tolist()
    matrix = build_distance_matrix(codes, distances, weights)
    family_order, family_indices = order_families(matrix, df["family"])
    family_lang_order = build_family_language_orders(matrix, df, family_order, family_indices)
    sequence = build_language_sequence(family_order, family_lang_order, args.block_size)

    prev_total = 0
    results = []
    for tier in tiers:
        best = search_tier_by_prefix(
            tier,
            sequence,
            df,
            matrix,
            alpha=args.alpha,
            beta=args.beta,
            prev_total=prev_total,
        )
        prev_total = best["total"]
        results.append({"tier": tier["name"], **best})
        print(
            f"{tier['name']}: total={best['total']} score={best['score']:.3f} inter={best['inter']:.3f} intra={best['intra']:.3f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # convert numpy types to python
    serializable = []
    for item in results:
        entry = {k: v for k, v in item.items() if k not in {"codes", "per_family"}}
        entry.update(
            {
                "codes": item["codes"],
                "per_family": item["per_family"],
            }
        )
        serializable.append(entry)
    output_path.write_text(json.dumps(serializable, indent=2))
    print(f"Saved tier definitions to {output_path}")


if __name__ == "__main__":
    main()
