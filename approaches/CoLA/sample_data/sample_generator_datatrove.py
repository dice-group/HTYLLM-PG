"""Sample FineWeb/FineWeb2 subsets based on a single CSV plan."""

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

PLAN_COLUMNS = ("subset",)


def parse_args():
    parser = argparse.ArgumentParser(description="Datatrove sampler driven by CSV input.")
    parser.add_argument("plan_csv", type=Path, help="CSV listing languages (requires `subset` column, optional `documents`).")
    parser.add_argument("max_sample", type=int, help="Maximum number of rows sampled per language.")
    parser.add_argument("output_dir", type=Path, help="Directory where sample JSONL files and manifest will be emitted.")
    return parser.parse_args()


def _load_plan(csv_path: Path, max_sample: int) -> Iterable[Tuple[str, int]]:
    df = pd.read_csv(csv_path)
    for col in PLAN_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing `{col}` column in {csv_path}.")
    for _, row in df.iterrows():
        subset = str(row["subset"]).strip()
        docs = int(row["documents"]) if "documents" in df.columns else max_sample
        docs = docs if docs > 0 else max_sample
        size = min(max_sample, docs)
        yield subset, size


def _build_executor(out_dir: Path, subset: str, sample_size: int) -> LocalPipelineExecutor:
    dataset = "hf://datasets/HuggingFaceFW/fineweb/data/" if subset.lower() == "english" else f"hf://datasets/HuggingFaceFW/fineweb-2/data/{subset}/train"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(str(out_dir))
    reader = ParquetReader(dataset, limit=sample_size, file_progress=True, doc_progress=True)
    return LocalPipelineExecutor(pipeline=[reader, writer], tasks=1)


def _write_manifest(output_dir: Path, entries):
    manifest = output_dir / "sample_sizes.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as fp:
        fp.write("subset\tsample_size\n")
        for subset, size in entries:
            fp.write(f"{subset}\t{size}\n")


def main() -> None:
    args = parse_args()
    plan_path = args.plan_csv.resolve()
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for subset, sample_size in _load_plan(plan_path, args.max_sample):
        manifest_rows.append((subset, sample_size))
        executor = _build_executor(output_dir / subset, subset, sample_size)
        executor.run()

    _write_manifest(output_dir, manifest_rows)


if __name__ == "__main__":
    main()
