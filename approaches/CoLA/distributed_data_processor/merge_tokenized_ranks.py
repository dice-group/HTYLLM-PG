import argparse
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datasets import load_from_disk, concatenate_datasets


def merge_pair(args):
    a, b, out = args
    da = load_from_disk(str(a))
    db = load_from_disk(str(b))
    merged = concatenate_datasets([da, db])
    merged.save_to_disk(str(out))
    return out


def discover_rank_dirs(root: Path) -> list[Path]:
    ranks = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("rank_"))
    if not ranks:
        raise RuntimeError("No rank_* dirs found")
    return ranks


def _count_stages(total: int) -> int:
    count = 0
    current = total
    while current > 1:
        count += 1
        current = (current // 2) + (current % 2)
    return max(count, 0)


def pairwise_merge_concurrent(dirs: list[Path], tmp_root: Path, max_workers: int = 4) -> Path:
    current = dirs
    stage = 0
    total_stages = _count_stages(len(current))

    total_initial = len(current)
    while len(current) > 1:
        stage += 1
        tasks = []
        next_round = []
        stage_dir = tmp_root / f"stage_{stage:03d}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(current), 2):
            if i + 1 == len(current):
                next_round.append(current[i])
                break

            a, b = current[i], current[i + 1]
            pair_idx = i // 2
            out = stage_dir / f"pair_{pair_idx:05d}"
            tasks.append((a, b, out))

        remaining_pct = (len(current) / total_initial) * 100
        if tasks:
            print(f"[merge][stage {stage}/{total_stages}] start: {len(tasks)} pairs, {len(current)} datasets ({remaining_pct:.1f}%).")
        completed = 0
        total_pairs = len(tasks)
        if tasks:
            if max_workers <= 1:
                for task in tasks:
                    out_dir = merge_pair(task)
                    completed += 1
                    pct_pairs = (completed / total_pairs) * 100 if total_pairs else 100
                    print(f"[merge][stage {stage}/{total_stages}] pair {completed}/{total_pairs} done ({pct_pairs:.1f}%).")
                    next_round.append(out_dir)
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    for out_dir in ex.map(merge_pair, tasks):
                        completed += 1
                        pct_pairs = (completed / total_pairs) * 100 if total_pairs else 100
                        print(f"[merge][stage {stage}/{total_stages}] pair {completed}/{total_pairs} done ({pct_pairs:.1f}%).")
                        next_round.append(out_dir)

        current = next_round
        remaining_pct = (len(current) / total_initial) * 100 if total_initial else 0
        print(f"[merge][stage {stage}/{total_stages}] end: {len(current)} datasets remain ({remaining_pct:.1f}%).")

    return current[0]


def merge_ranks(root: Path, output: Path, overwrite: bool, workers: int):
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"{output} exists. Use --overwrite.")
        shutil.rmtree(output)

    ranks = discover_rank_dirs(root)

    with tempfile.TemporaryDirectory() as tmp:
        final_dir = pairwise_merge_concurrent(ranks, Path(tmp), workers)
        shutil.copytree(final_dir, output)

    print(f"Merged dataset saved to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized_root", type=Path, required=True)
    ap.add_argument("--output_path", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    merge_ranks(args.tokenized_root, args.output_path, args.overwrite, args.workers)


if __name__ == "__main__":
    main()
