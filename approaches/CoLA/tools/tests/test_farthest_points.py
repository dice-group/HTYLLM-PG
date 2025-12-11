import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.farthest_points import assign_neighbors, select_farthest_points

_SUBSET_LOOKUP: dict[str, str] | None = None


def _load_subset_lookup() -> dict[str, str]:
    global _SUBSET_LOOKUP
    if _SUBSET_LOOKUP is not None:
        return _SUBSET_LOOKUP
    from tools.farthest_points import load_subset_map

    inverted: dict[str, str] = {}
    for code, subset in load_subset_map().items():
        if subset and subset not in inverted:
            inverted[subset] = code
    _SUBSET_LOOKUP = inverted
    return inverted


def _subset_to_code(label: str) -> str:
    lookup = _load_subset_lookup()
    return lookup.get(label, label)


def line_distance(positions: list[float]) -> np.ndarray:
    coords = np.array(positions, dtype=float)
    return np.abs(coords[:, None] - coords[None, :])


def test_select_farthest_pair():
    dist = line_distance([0.0, 1.0, 2.0, 12.0])
    indices = set(select_farthest_points(dist, k=2))
    assert indices == {0, 3}


def test_select_farthest_three_points():
    dist = line_distance([0.0, 2.0, 5.0, 9.0])
    indices = set(select_farthest_points(dist, k=3))
    assert indices == {0, 2, 3}


def test_select_single_point_prefers_extreme():
    dist = line_distance([0.0, 1.0, 3.0])
    idx = select_farthest_points(dist, k=1)
    assert idx == [2], "point with largest total distance should be chosen"


def test_embed_and_plot(tmp_path: Path):
    dist = line_distance([0.0, 2.0, 6.0])
    codes = np.array(["a", "b", "c"], dtype="<U1")
    npz_path = tmp_path / "distances.npz"
    np.savez(npz_path, matrix=dist, codes=codes)

    out_png = tmp_path / "plot.png"
    out_json = tmp_path / "codes.json"

    # Run via CLI to ensure plotting + JSON output works.
    import subprocess
    import sys

    base_cmd = [
        sys.executable,
        "tools/farthest_points.py",
        "--distance-npz",
        str(npz_path),
        "--k",
        "2",
        "--target-total",
        "3",
        "--output-image",
        str(out_png),
        "--json-output",
        str(out_json),
    ]
    subprocess.check_call(base_cmd)
    subprocess.check_call(base_cmd + ["--show-all"])

    assert out_png.exists()
    data = json.loads(out_json.read_text())
    assert set(data["best_codes"]) == {"a", "c"}
    assert data["best_neighbors"]["a"] == ["b"]
    assert data["best_neighbors"]["c"] == []
    qm = data["quality_metrics"]
    assert qm["farthest_pairwise"]["min"] > 0
    assert qm["neighbor_coherence"]["summary"]["mean"] >= 0
    assert len(data["per_k"]) >= 1


def test_fps_basic_properties():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    idx = select_farthest_points(D, k=5)
    assert len(idx) == 5
    assert len(set(idx)) == 5
    assert all(0 <= i < 20 for i in idx)


def test_fps_k_greater_than_n():
    D = np.zeros((3, 3))
    idx = select_farthest_points(D, k=10)
    assert len(idx) == 3
    assert len(set(idx)) == 3


def test_fps_rejects_non_square():
    with pytest.raises(ValueError):
        select_farthest_points(np.zeros((3, 4)), k=2)


def test_fps_rejects_negative():
    D = np.zeros((3, 3))
    D[0, 1] = -1
    with pytest.raises(ValueError):
        select_farthest_points(D, k=2)


def test_fps_rejects_nan():
    D = np.zeros((3, 3))
    D[0, 1] = np.nan
    with pytest.raises(ValueError):
        select_farthest_points(D, k=2)


def min_pairwise(D: np.ndarray, idx: list[int]) -> float:
    sub = D[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, np.inf)
    return float(np.min(sub))


def test_fps_beats_random_on_average():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    k = 8
    fps_idx = select_farthest_points(D, k=k)
    fps_score = min_pairwise(D, fps_idx)

    random_scores = []
    for _ in range(200):
        r = rng.choice(50, size=k, replace=False).tolist()
        random_scores.append(min_pairwise(D, r))

    assert fps_score >= np.percentile(random_scores, 75)


def test_compute_neighbors_basic():
    dist = line_distance([0.0, 2.0, 6.0, 10.0])
    neighbors = assign_neighbors(dist, [0, 3], [1, 1], total_neighbors=2)
    assert neighbors[0] == [1]
    assert neighbors[3] == [2]


def test_quality_metrics_values(tmp_path: Path):
    positions = [0.0, 5.0, 15.0, 20.0]
    dist = line_distance(positions)
    codes = np.array(["l0", "l1", "l2", "l3"])
    npz_path = tmp_path / "line.npz"
    np.savez(npz_path, matrix=dist, codes=codes)
    out_json = tmp_path / "metrics.json"

    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "tools/farthest_points.py",
            "--distance-npz",
            str(npz_path),
            "--k",
            "2",
            "--target-total",
            "4",
            "--json-output",
            str(out_json),
        ]
    )

    data = json.loads(out_json.read_text())
    pairwise = data["quality_metrics"]["farthest_pairwise"]
    assert pairwise["min"] > 0
    neighbor_summary = data["quality_metrics"]["neighbor_coherence"]["summary"]
    assert neighbor_summary["mean"] > 0


def test_k_range_selects_best(tmp_path: Path):
    dist = line_distance([0.0, 2.0, 4.0, 6.0, 100.0, 102.0, 104.0])
    codes = np.array(["a", "b", "c", "d", "e", "f", "g"])
    npz_path = tmp_path / "points.npz"
    np.savez(npz_path, matrix=dist, codes=codes)
    out_json = tmp_path / "range.json"

    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "tools/farthest_points.py",
            "--distance-npz",
            str(npz_path),
            "--k-min",
            "2",
            "--k-max",
            "3",
            "--target-total",
            "7",
            "--json-output",
            str(out_json),
        ]
    )

    data = json.loads(out_json.read_text())
    assert data["best_k"] in (2, 3)
    scores = {entry["k"]: entry["score"] for entry in data["per_k"]}
    assert data["best_k"] == max(scores, key=scores.get)


def test_limit_languages_flag(tmp_path: Path):
    dist = line_distance([0.0, 1.0, 3.0, 7.0])
    codes = np.array(["a", "b", "c", "d"], dtype="<U1")
    npz_path = tmp_path / "distances.npz"
    np.savez(npz_path, matrix=dist, codes=codes)

    out_json = tmp_path / "result.json"

    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "tools/farthest_points.py",
            "--distance-npz",
            str(npz_path),
            "--k",
            "2",
            "--target-total",
            "3",
            "--json-output",
            str(out_json),
            "--limit-languages",
            "3",
        ]
    )

    data = json.loads(out_json.read_text())
    assert set(data["best_codes"]).issubset({"a", "b", "c"})


def test_coordinate_csv_mode(tmp_path: Path):
    size = 15  # 225 points
    points = []
    codes = []
    for i in range(size):
        for j in range(size):
            codes.append(f"p{i:02d}_{j:02d}")
            points.append((float(i), float(j)))
    points = np.asarray(points, dtype=float)
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    npz_path = tmp_path / "grid.npz"
    csv_path = tmp_path / "coords.csv"
    np.savez(npz_path, matrix=dist, codes=np.array(codes, dtype="<U16"))
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("code,x,y\n")
        for code, (x, y) in zip(codes, points):
            f.write(f"{code},{x},{y}\n")

    out_png = tmp_path / "plot.png"
    out_json = tmp_path / "result.json"

    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "tools/farthest_points.py",
            "--distance-npz",
            str(npz_path),
            "--coordinate-csv",
            str(csv_path),
            "--k",
            "4",
            "--neighbors-per-language",
            "1",
            "--output-image",
            str(out_png),
            "--json-output",
            str(out_json),
            "--show-all",
        ]
    )

    assert out_png.exists()
    data = json.loads(out_json.read_text())
    expected_corners = {
        "p00_00",
        f"p00_{size-1:02d}",
        f"p{size-1:02d}_00",
        f"p{size-1:02d}_{size-1:02d}",
    }
    assert set(data["best_codes"]) == expected_corners


def test_cli_lang2vec_distance(tmp_path: Path):
    meta_path = tmp_path / "meta.csv"
    meta_path.write_text("code,documents\neng,100\nspa,80\nfra,40\n", encoding="utf-8")
    out_png = tmp_path / "lang2vec.png"
    out_json = tmp_path / "lang2vec.json"

    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "tools/farthest_points.py",
            "--lang2vec-distance",
            "genetic",
            "--metadata-csv",
            str(meta_path),
            "--min-documents",
            "50",
            "--k",
            "2",
            "--target-total",
            "2",
            "--output-image",
            str(out_png),
            "--json-output",
            str(out_json),
        ]
    )

    data = json.loads(out_json.read_text())
    assert data["best_k"] == 2
    assert set(data["best_codes"]).issubset({"english", "spa_Latn"})


def test_neighbors_are_closest_to_their_seed(tmp_path: Path):
    # small synthetic matrix where each language has a unique closest seed
    codes = np.array(["s1", "s2", "n11", "n12", "n21", "n22"])
    dist = np.array(
        [
            [0, 5, 1, 2, 8, 9],
            [5, 0, 8, 9, 1, 2],
            [1, 8, 0, 1.5, 7, 8],
            [2, 9, 1.5, 0, 7.5, 8.5],
            [8, 1, 7, 7.5, 0, 1],
            [9, 2, 8, 8.5, 1, 0],
        ]
    )
    npz_path = tmp_path / "synthetic.npz"
    np.savez(npz_path, matrix=dist, codes=codes)
    out_json = tmp_path / "result.json"

    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "tools/farthest_points.py",
            "--distance-npz",
            str(npz_path),
            "--k",
            "2",
            "--target-total",
            "6",
            "--json-output",
            str(out_json),
        ]
    )

    data = json.loads(out_json.read_text())
    seed_indices = {code: idx for idx, code in enumerate(codes)}
    seeds = data["best_codes"]
    matrix = dist

    for seed in seeds:
        seed_idx = seed_indices[seed]
        for neighbor in data["best_neighbors"][seed]:
            neighbor_idx = seed_indices[neighbor]
            assigned = matrix[neighbor_idx, seed_idx]
            for other in seeds:
                if other == seed:
                    continue
                other_idx = seed_indices[other]
                assert assigned <= matrix[neighbor_idx, other_idx]


def _verify_neighbors(output_json: Path, matrix: np.ndarray, codes: list[str]) -> None:
    data = json.loads(output_json.read_text())
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    seeds = data["best_codes"]
    for seed in seeds:
        seed_code = _subset_to_code(seed)
        if seed_code not in code_to_idx:
            continue
        seed_idx = code_to_idx[seed_code]
        for neighbor in data["best_neighbors"][seed]:
            neighbor_code = _subset_to_code(neighbor)
            if neighbor_code not in code_to_idx:
                continue
            neighbor_idx = code_to_idx[neighbor_code]
            assigned = matrix[neighbor_idx, seed_idx]
            diffs = [
                matrix[neighbor_idx, code_to_idx[_subset_to_code(other)]] - assigned
                for other in seeds
                if other != seed
                and _subset_to_code(other) in code_to_idx
            ]
            assert min(diffs) >= -0.12


def test_real_data_neighbors_align(tmp_path: Path):
    npz_path = Path("data_prep/processed_artifacts/lang2vec_all_distances.npz")
    metadata_path = Path("data_prep/base_data/fineweb2-language-distribution.csv")
    if not npz_path.exists() or not metadata_path.exists():
        pytest.skip("Real data artifacts missing; skip integration check.")

    matrix = np.load(npz_path)["matrix"].astype(float)
    codes = [str(code) for code in np.load(npz_path)["codes"]]

    import subprocess
    import sys

    for total in (72, 200):
        out_json = tmp_path / f"real_alignment_{total}.json"
        subprocess.check_call(
            [
                sys.executable,
                "tools/farthest_points.py",
                "--distance-npz",
                str(npz_path),
                "--k-min",
                "4",
                "--k-max",
                "16",
                "--target-total",
                str(total),
                "--limit-languages",
                "300",
                "--metadata-csv",
                str(metadata_path),
                "--min-documents",
                "500",
                "--json-output",
                str(out_json),
            ]
        )
        _verify_neighbors(out_json, matrix, codes)
