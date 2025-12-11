from __future__ import annotations
import argparse
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Sequence

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))

from tokenizer_extension.allocation import AllocationConfig, compute_allocation
from tokenizer_extension.coverage import CoverageConfig, compute_coverage
from tokenizer_extension.extension import ExtensionConfig, ExtensionResult, extend_tokenizer
from tokenizer_extension.training import TrainingConfig, TrainingResult, train_tokenizer
from tokenizer_extension.utils import load_pipeline_config, compare_metrics


@dataclass
class ComparisonResult:
    dataframe: pd.DataFrame
    output_csv: Optional[Path]

@dataclass
class PipelineConfig:
    output_dir: Path
    allocation: AllocationConfig
    training: Optional[TrainingConfig] = None
    base_coverage: CoverageConfig | None = None
    extension: Optional[ExtensionConfig] = None
    extended_coverage: Optional[CoverageConfig] = None

@dataclass
class PipelineResult:
    base_metrics: pd.DataFrame
    allocation: pd.DataFrame
    allocation_csv: Path
    training: Optional[TrainingResult]
    extension: Optional[ExtensionResult]
    extended_metrics: Optional[pd.DataFrame]
    comparison: Optional[ComparisonResult]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. always compute base tokenizer metrics
    base_metrics_path = out_dir / "base_metrics.csv"
    config.base_coverage.output_csv = base_metrics_path
    base_metrics_df = compute_coverage(config.base_coverage)
    print(f"[pipeline] Base metrics saved to {base_metrics_path}")

    # 2. compute allocation
    allocation_df = compute_allocation(base_metrics_df, config.allocation)
    allocation_path = out_dir / "allocation.csv"
    allocation_df.to_csv(allocation_path, index=False)
    print(f"[pipeline] Allocation saved to {allocation_path}")

    # 3. optional: train tokenizer
    training_result = train_tokenizer(config.training) if config.training else None

    # 4. optional: extend tokenizer
    extension_result = (
        extend_tokenizer(config.extension, allocation_df) if config.extension else None
    )

    # 5. optional: compute extended coverage
    extended_metrics_df = None
    if config.extended_coverage:
        extended_metrics_path = out_dir / "extended_metrics.csv"
        config.extended_coverage.output_csv = extended_metrics_path
        extended_metrics_df = compute_coverage(config.extended_coverage)
        print(f"[pipeline] Extended metrics saved to {extended_metrics_path}")

    # 6. compare metrics
    comparison_result = None
    if extended_metrics_df is not None:
        comparison_df = compare_metrics(base_metrics_df, extended_metrics_df)
        comparison_path = out_dir / "comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        comparison_result = ComparisonResult(comparison_df, comparison_path)
        print(f"[pipeline] Comparison saved to {comparison_path}")

    return PipelineResult(
        base_metrics=base_metrics_df,
        allocation=allocation_df,
        allocation_csv=allocation_path,
        training=training_result,
        extension=extension_result,
        extended_metrics=extended_metrics_df,
        comparison=comparison_result,
    )

def main(argv: Optional[Sequence[str]] = None) -> Optional[PipelineResult]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="path to config file")
    config_paths = parser.parse_args(argv).config

    last_result: Optional[PipelineResult] = None
    for path in config_paths:
        parts = load_pipeline_config(path)
        pipeline_config = PipelineConfig(**parts)
        print(f"[tokenize_extension] Running config {path}")
        last_result = run_pipeline(pipeline_config)
    return last_result

if __name__ == "__main__":
    main()
