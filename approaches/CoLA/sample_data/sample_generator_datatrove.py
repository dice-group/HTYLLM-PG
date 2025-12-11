"""Sample FineWeb/FineWeb2 subsets based on a single CSV plan."""

import argparse
import fcntl
import math
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

PLAN_COLUMNS = ("subset",)


def _int_arg(value: str) -> int:
    return int(value.replace("_", ""))


def parse_args():
    parser = argparse.ArgumentParser(description="Datatrove sampler driven by CSV input.")
    parser.add_argument("plan_csv", type=Path, help="CSV listing languages (requires `subset` column, optional `documents`).")
    parser.add_argument("output_dir", type=Path, help="Directory where sample JSONL files and manifest will be emitted.")
    parser.add_argument("--max-sample", type=_int_arg, help="Optional cap on rows sampled per language.")
    parser.add_argument("--tokenizer-training", action="store_true", help="If set, only sample 5%% of the plan allocations for tokenizer training.")
    parser.add_argument("--shard-index", type=int, help="Index of this shard when splitting the CSV across array jobs.")
    parser.add_argument("--num-shards", type=int, help="Total number of shards. Defaults to Slurm array size or 1.")
    return parser.parse_args()

def _resolve_shard(args) -> Tuple[int, int]:
    shard_idx = args.shard_index if args.shard_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_shards = args.num_shards if args.num_shards is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if shard_idx < 0:
        raise ValueError("Shard index must be >= 0.")
    if num_shards <= 0:
        raise ValueError("Number of shards must be >= 1.")
    if shard_idx >= num_shards:
        raise ValueError(f"Shard index {shard_idx} out of range for {num_shards} shards.")
    return shard_idx, num_shards


def _load_plan(csv_path: Path, max_sample: int | None, tokenizer_training: bool) -> List[Tuple[str, int]]:
    df = pd.read_csv(csv_path)
    for col in PLAN_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing `{col}` column in {csv_path}.")
    plan = []
    for _, row in df.iterrows():
        subset = str(row["subset"]).strip()
        docs = int(row["documents"]) if "documents" in df.columns else max_sample
        docs = docs if docs and docs > 0 else max_sample
        size = min(max_sample, docs) if max_sample else docs
        if tokenizer_training and size:
            size = max(1, int(size * 0.05))
        plan.append((subset, size))
    return plan


def _slice_plan(plan: Sequence[Tuple[str, int]], shard_idx: int, num_shards: int):
    total = len(plan)
    if total == 0:
        return []
    chunk = math.ceil(total / num_shards)
    start = shard_idx * chunk
    end = min(start + chunk, total)
    return plan[start:end]


def _build_executor(root_dir: Path, subset: str, sample_size: int) -> LocalPipelineExecutor:
    dataset = "hf://datasets/HuggingFaceFW/fineweb/data/" if subset.lower() in ("english", "eng_latn") else f"hf://datasets/HuggingFaceFW/fineweb-2/data/{subset}/train"
    out_dir = root_dir / "samples" / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(str(out_dir))
    reader = ParquetReader(dataset, limit=sample_size, file_progress=True, doc_progress=True)
    return LocalPipelineExecutor(pipeline=[reader, writer], tasks=1)


def _append_manifest(root_dir: Path, entries):
    manifest = root_dir / "sample_sizes.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    with os.fdopen(fd, "a", encoding="utf-8") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        current_size = os.stat(fp.fileno()).st_size
        if current_size == 0:
            fp.write("subset\tsample_size\n")
        for subset, sample_size in entries:
            fp.write(f"{subset}\t{sample_size}\n")
        fp.flush()
        os.fsync(fp.fileno())
        fcntl.flock(fp, fcntl.LOCK_UN)


def main() -> None:
    args = parse_args()
    plan_path = args.plan_csv.resolve()
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx, num_shards = _resolve_shard(args)
    full_plan = _load_plan(plan_path, args.max_sample, args.tokenizer_training)
    shard_plan = _slice_plan(full_plan, shard_idx, num_shards)

    manifest_rows = []
    for subset, sample_size in shard_plan:
        manifest_rows.append((subset, sample_size))
        executor = _build_executor(output_dir, subset, sample_size)
        executor.run()

    if manifest_rows:
        _append_manifest(output_dir, manifest_rows)


if __name__ == "__main__":
    main()
