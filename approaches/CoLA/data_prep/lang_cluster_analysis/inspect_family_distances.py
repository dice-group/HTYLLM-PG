import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
FW_LANG_PATH = ROOT / "data_prep" / "base_data" / "fineweb2-language-distribution.csv"
LANG2VEC_DIR = HERE.parent / "lang2vec"
sys.path.insert(0, str(LANG2VEC_DIR))
from lang2vec import lang2vec as l2v  # noqa: E402


def available_distance_langs() -> set[str]:
    try:
        return set(l2v.DISTANCE_LANGUAGES)
    except AttributeError:
        return set(l2v.available_distance_languages())


def load_overlap(limit: int | None) -> pd.DataFrame:
    fw = pd.read_csv(FW_LANG_PATH, usecols=["code", "family"]).drop_duplicates("code")
    overlap = fw[fw["code"].isin(available_distance_langs())]
    if limit and limit > 0:
        overlap = overlap.head(limit)
    return overlap.reset_index(drop=True)


def build_distance_matrix(distance_type: str, overlap: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    langs = overlap["code"].tolist()
    fams = overlap["family"].tolist()
    matrix = np.asarray(l2v.distance(distance_type, langs), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return matrix, langs, fams


def family_index_map(families: List[str]) -> Dict[str, List[int]]:
    idx = defaultdict(list)
    for pos, fam in enumerate(families):
        idx[fam].append(pos)
    return idx


def compute_family_medoids(matrix: np.ndarray, fam_idx: Dict[str, List[int]]):
    families = sorted(fam_idx)
    medoid_indices: Dict[str, int] = {}
    for fam in families:
        idxs = fam_idx[fam]
        if len(idxs) == 1:
            medoid_indices[fam] = idxs[0]
            continue
        sub = matrix[np.ix_(idxs, idxs)]
        mean_dists = sub.mean(axis=1)
        medoid_indices[fam] = idxs[int(np.argmin(mean_dists))]
    medoid_matrix = np.zeros((len(families), len(families)), dtype=float)
    for i, fam_a in enumerate(families):
        for j in range(i + 1, len(families)):
            fam_b = families[j]
            dist = float(matrix[medoid_indices[fam_a], medoid_indices[fam_b]])
            medoid_matrix[i, j] = medoid_matrix[j, i] = dist
    return families, medoid_indices, medoid_matrix


def greedy_select_families(candidates: List[str], k: int, F_medoid: np.ndarray, fam_order: Dict[str, int]) -> List[str]:
    if len(candidates) <= k:
        return candidates
    indices = [fam_order[fam] for fam in candidates]
    sub = F_medoid[np.ix_(indices, indices)]
    selected = [0]
    while len(selected) < k:
        remaining = [i for i in range(len(candidates)) if i not in selected]
        if not remaining:
            break
        pick = max(remaining, key=lambda idx: min(sub[idx, s] for s in selected))
        selected.append(pick)
    return [candidates[i] for i in selected]


def select_languages_for_family(matrix: np.ndarray, langs: List[str], idxs: List[int], per_family: int) -> List[int]:
    if len(idxs) <= per_family:
        return idxs[:]
    sub = matrix[np.ix_(idxs, idxs)]
    work = sub.copy()
    np.fill_diagonal(work, np.inf)
    first_flat = int(np.argmin(work))
    size = work.shape[0]
    first_i, first_j = divmod(first_flat, size)
    selected = [first_i, first_j]
    while len(selected) < per_family:
        remaining = [i for i in range(size) if i not in selected]
        best = min(remaining, key=lambda r: sub[r, selected].mean())
        selected.append(best)
    return [idxs[i] for i in selected]


def mean_intra_family_distance(matrix: np.ndarray, indices: List[int]) -> float:
    if len(indices) < 2:
        return 0.0
    sub = matrix[np.ix_(indices, indices)]
    tri = np.triu_indices(len(indices), k=1)
    return float(sub[tri].mean())


def mean_inter_family_distance_from_selection(matrix: np.ndarray, lang_selection: Dict[str, List[int]]) -> float:
    families = list(lang_selection)
    if len(families) < 2:
        return 0.0
    values = []
    for i, fam_a in enumerate(families):
        idx_a = lang_selection[fam_a]
        for fam_b in families[i + 1 :]:
            idx_b = lang_selection[fam_b]
            block = matrix[np.ix_(idx_a, idx_b)]
            values.append(float(block.mean()))
    return float(np.mean(values))


def evaluate_combo(
    matrix: np.ndarray,
    langs: List[str],
    fam_idx: Dict[str, List[int]],
    families_sorted: List[str],
    F_medoid: np.ndarray,
    fam_order: Dict[str, int],
    num_families: int,
    per_family: int,
    alpha: float,
    beta: float,
) -> tuple[float, Dict[str, List[str]], float, float] | None:
    candidates = [fam for fam in families_sorted if len(fam_idx[fam]) >= per_family]
    if len(candidates) < num_families or num_families < 2:
        return None
    selected_families = greedy_select_families(candidates, num_families, F_medoid, fam_order)
    lang_selection: Dict[str, List[int]] = {}
    for fam in selected_families:
        chosen = select_languages_for_family(matrix, langs, fam_idx[fam], per_family)
        lang_selection[fam] = chosen
    inter = mean_inter_family_distance_from_selection(matrix, lang_selection)
    intra = np.mean([mean_intra_family_distance(matrix, idxs) for idxs in lang_selection.values()])
    score = alpha * inter - beta * intra
    selected_lang_codes = {fam: [langs[i] for i in idxs] for fam, idxs in lang_selection.items()}
    return score, selected_lang_codes, inter, intra


def search_best_configuration(
    matrix: np.ndarray,
    langs: List[str],
    fam_idx: Dict[str, List[int]],
    families_sorted: List[str],
    F_medoid: np.ndarray,
    fam_order: Dict[str, int],
    min_total: int,
    max_total: int,
    min_families: int,
    max_families: int,
    min_per_family: int,
    max_per_family: int,
    alpha: float,
    beta: float,
) -> Tuple[dict | None, List[dict]]:
    best = None
    combos: List[dict] = []
    for num_families in range(min_families, min(max_families, len(families_sorted)) + 1):
        for per_family in range(min_per_family, max_per_family + 1):
            total = num_families * per_family
            if total < min_total or total > max_total:
                continue
            result = evaluate_combo(
                matrix,
                langs,
                fam_idx,
                families_sorted,
                F_medoid,
                fam_order,
                num_families,
                per_family,
                alpha,
                beta,
            )
            if result is None:
                continue
            score, lang_map, inter, intra = result
            combos.append(
                {
                    "families": num_families,
                    "per_family": per_family,
                    "total": total,
                    "score": score,
                    "inter": inter,
                    "intra": intra,
                }
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "families": num_families,
                    "per_family": per_family,
                    "total_langs": num_families * per_family,
                    "lang_map": lang_map,
                    "inter": inter,
                    "intra": intra,
                }
    return best, combos


def main():
    parser = argparse.ArgumentParser(
        description="Search for the best (families, languages-per-family) combo balancing separation and tightness."
    )
    parser.add_argument("--distance", default="genetic", choices=l2v.DISTANCES)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on overlapping languages considered.")
    parser.add_argument("--min-total", type=int, default=72)
    parser.add_argument("--max-total", type=int, default=95)
    parser.add_argument("--min-families", type=int, default=2)
    parser.add_argument("--max-families", type=int, default=25)
    parser.add_argument("--min-per-family", type=int, default=2)
    parser.add_argument("--max-per-family", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for inter-family distance.")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for intra-family distance.")
    args = parser.parse_args()

    overlap = load_overlap(args.limit if args.limit and args.limit > 0 else None)
    if len(overlap) < 2:
        raise SystemExit("Need at least two overlapping languages.")
    matrix, langs, fams = build_distance_matrix(args.distance, overlap)
    fam_idx = family_index_map(fams)
    families_sorted, medoid_map, F_medoid = compute_family_medoids(matrix, fam_idx)
    fam_order = {fam: idx for idx, fam in enumerate(families_sorted)}

    best, combos = search_best_configuration(
        matrix,
        langs,
        fam_idx,
        families_sorted,
        F_medoid,
        fam_order,
        args.min_total,
        args.max_total,
        args.min_families,
        args.max_families,
        args.min_per_family,
                args.max_per_family,
                args.alpha,
                args.beta,
            )
    if not best:
        raise SystemExit("No combination satisfied the constraints.")

    print(
        f"Best combo: {best['families']} families × {best['per_family']} langs "
        f"= {best['total_langs']} languages (score={best['score']:.3f}, "
        f"inter={best['inter']:.3f}, intra={best['intra']:.3f})"
    )
    if combos:
        print("\nAll viable combos:")
        for entry in sorted(combos, key=lambda d: -d["score"]):
            print(
                f"  {entry['families']}×{entry['per_family']} "
                f"(total={entry['total']}, score={entry['score']:.3f}, "
                f"inter={entry['inter']:.3f}, intra={entry['intra']:.3f})"
            )
    for fam, codes in best["lang_map"].items():
        print(f"\n{fam} ({len(codes)} langs): {', '.join(codes)}")


if __name__ == "__main__":
    main()
