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
    base_metrics: Optional[pd.DataFrame]
    allocation: Optional[pd.DataFrame]
    allocation_csv: Path
    training: Optional[TrainingResult]
    extension: Optional[ExtensionResult]
    extended_metrics: Optional[pd.DataFrame]
    comparison: Optional[ComparisonResult]


STAGE_ORDER = (
    "base_coverage",
    "allocation",
    "training",
    "extension",
    "extended_coverage",
    "comparison",
)


def _resolve_stages(stages: Optional[Sequence[str]]) -> list[str]:
    if stages is None:
        return list(STAGE_ORDER)

    invalid = [stage for stage in stages if stage not in STAGE_ORDER]
    if invalid:
        raise ValueError(f"Unknown stage(s): {', '.join(invalid)}")

    selected = set(stages)
    return [stage for stage in STAGE_ORDER if stage in selected]


def run_base_coverage(config: PipelineConfig) -> pd.DataFrame:
    if config.base_coverage is None:
        raise ValueError("Base coverage configuration is missing.")

    base_metrics_path = config.output_dir / "base_metrics.csv"
    cov_cfg = config.base_coverage
    print(
        "[pipeline] base_coverage starting\n"
        f"  data_dir    : {cov_cfg.data_dir}\n"
        f"  tokenizer   : {cov_cfg.tokenizer}\n"
        f"  max_lines   : {cov_cfg.max_lines}\n"
        f"  num_workers : {cov_cfg.num_workers or 'auto'}\n"
        f"  output_csv  : {base_metrics_path}"
    )
    config.base_coverage.output_csv = base_metrics_path
    df = compute_coverage(config.base_coverage)
    print(f"[pipeline] Base metrics saved to {base_metrics_path}")
    return df


def run_allocation(
    config: PipelineConfig,
    base_metrics: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_metrics_df = base_metrics
    if base_metrics_df is None:
        base_metrics_path = config.output_dir / "base_metrics.csv"
        if not base_metrics_path.exists():
            raise FileNotFoundError(
                f"Base metrics not found at {base_metrics_path}. "
                "Run the base_coverage stage first."
            )
        print(f"[pipeline] allocation: loading base metrics from {base_metrics_path}")
        base_metrics_df = pd.read_csv(base_metrics_path)

    allocation_df = compute_allocation(base_metrics_df, config.allocation)
    allocation_path = config.output_dir / "allocation.csv"
    allocation_df.to_csv(allocation_path, index=False)
    print(
        "[pipeline] allocation summary\n"
        f"  languages        : {len(allocation_df)}\n"
        f"  total_new_tokens : {config.allocation.total_new_tokens}\n"
        f"  output_csv       : {allocation_path}"
    )
    print(f"[pipeline] Allocation saved to {allocation_path}")
    return allocation_df, base_metrics_df


def run_training_stage(config: PipelineConfig) -> TrainingResult:
    if config.training is None:
        raise ValueError("Training configuration is missing.")
    train_cfg = config.training
    print(
        "[pipeline] training starting\n"
        f"  data_dir    : {train_cfg.data_dir}\n"
        f"  output_dir  : {train_cfg.output_dir}\n"
        f"  base_model  : {train_cfg.base_model}\n"
        f"  vocab_size  : {train_cfg.vocab_size}\n"
        f"  max_samples : {train_cfg.max_samples or 'all'}"
    )
    return train_tokenizer(config.training)


def run_extension_stage(
    config: PipelineConfig,
    allocation: Optional[pd.DataFrame],
) -> ExtensionResult:
    if config.extension is None:
        raise ValueError("Extension configuration is missing.")

    allocation_df = allocation
    if allocation_df is None:
        allocation_path = config.output_dir / "allocation.csv"
        if not allocation_path.exists():
            raise FileNotFoundError(
                f"Allocation results not found at {allocation_path}. "
                "Run the allocation stage first."
            )
        print(f"[pipeline] extension: loading allocation from {allocation_path}")
        allocation_df = pd.read_csv(allocation_path)

    ext_cfg = config.extension
    print(
        "[pipeline] extension starting\n"
        f"  base_tokenizer       : {ext_cfg.base_tokenizer_path}\n"
        f"  multilingual_source  : {ext_cfg.multilingual_tokenizer_path}\n"
        f"  data_dir             : {ext_cfg.data_dir}\n"
        f"  output_dir           : {ext_cfg.output_dir}\n"
        f"  vocab_cap            : {ext_cfg.vocab_cap or 'none'}\n"
        f"  sample_docs          : {ext_cfg.sample_docs}\n"
        f"  allocation_languages : {allocation_df.shape[0]}"
    )
    return extend_tokenizer(config.extension, allocation_df)


def run_extended_coverage(config: PipelineConfig) -> pd.DataFrame:
    if config.extended_coverage is None:
        raise ValueError("Extended coverage configuration is missing.")

    extended_metrics_path = config.output_dir / "extended_metrics.csv"
    ext_cov = config.extended_coverage
    print(
        "[pipeline] extended_coverage starting\n"
        f"  data_dir    : {ext_cov.data_dir}\n"
        f"  tokenizer   : {ext_cov.tokenizer}\n"
        f"  max_lines   : {ext_cov.max_lines}\n"
        f"  num_workers : {ext_cov.num_workers or 'auto'}\n"
        f"  output_csv  : {extended_metrics_path}"
    )
    config.extended_coverage.output_csv = extended_metrics_path
    df = compute_coverage(config.extended_coverage)
    print(f"[pipeline] Extended metrics saved to {extended_metrics_path}")
    return df


def run_comparison_stage(
    config: PipelineConfig,
    base_metrics: Optional[pd.DataFrame],
    extended_metrics: Optional[pd.DataFrame],
) -> ComparisonResult:
    base_df = base_metrics
    if base_df is None:
        base_metrics_path = config.output_dir / "base_metrics.csv"
        if not base_metrics_path.exists():
            raise FileNotFoundError(
                f"Base metrics not found at {base_metrics_path}. "
                "Run the base_coverage stage first."
            )
        print(f"[pipeline] comparison: loading base metrics from {base_metrics_path}")
        base_df = pd.read_csv(base_metrics_path)

    extended_df = extended_metrics
    if extended_df is None:
        extended_metrics_path = config.output_dir / "extended_metrics.csv"
        if not extended_metrics_path.exists():
            raise FileNotFoundError(
                f"Extended metrics not found at {extended_metrics_path}. "
                "Run the extended_coverage stage first."
            )
        print(f"[pipeline] comparison: loading extended metrics from {extended_metrics_path}")
        extended_df = pd.read_csv(extended_metrics_path)

    comparison_path = config.output_dir / "comparison.csv"
    print(
        "[pipeline] comparison starting\n"
        f"  base_metrics_rows    : {len(base_df)}\n"
        f"  extended_metrics_rows: {len(extended_df)}\n"
        f"  output_csv           : {comparison_path}"
    )
    comparison_df = compare_metrics(base_df, extended_df)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[pipeline] Comparison saved to {comparison_path}")

    return ComparisonResult(comparison_df, comparison_path)


def run_pipeline(
    config: PipelineConfig,
    stages: Optional[Sequence[str]] = None,
) -> PipelineResult:
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_stages = _resolve_stages(stages)
    print("[pipeline] execution plan:", ", ".join(resolved_stages))

    base_metrics_df: Optional[pd.DataFrame] = None
    allocation_df: Optional[pd.DataFrame] = None
    training_result: Optional[TrainingResult] = None
    extension_result: Optional[ExtensionResult] = None
    extended_metrics_df: Optional[pd.DataFrame] = None
    comparison_result: Optional[ComparisonResult] = None

    for stage in STAGE_ORDER:
        if stage not in resolved_stages:
            continue
        print(f"[pipeline] >>> entering stage '{stage}'")

        if stage == "base_coverage":
            base_metrics_df = run_base_coverage(config)
        elif stage == "allocation":
            allocation_df, base_metrics_df = run_allocation(config, base_metrics_df)
        elif stage == "training":
            if config.training is None:
                print("[pipeline] Training skipped (no configuration).")
                continue
            training_result = run_training_stage(config)
        elif stage == "extension":
            if config.extension is None:
                print("[pipeline] Extension skipped (no configuration).")
                continue
            allocation_for_extension = allocation_df
            extension_result = run_extension_stage(config, allocation_for_extension)
        elif stage == "extended_coverage":
            if config.extended_coverage is None:
                print("[pipeline] Extended coverage skipped (no configuration).")
                continue
            extended_metrics_df = run_extended_coverage(config)
        elif stage == "comparison":
            comparison_result = run_comparison_stage(
                config,
                base_metrics=base_metrics_df,
                extended_metrics=extended_metrics_df,
            )

        print(f"[pipeline] <<< finished stage '{stage}'")

    return PipelineResult(
        base_metrics=base_metrics_df,
        allocation=allocation_df,
        allocation_csv=out_dir / "allocation.csv",
        training=training_result,
        extension=extension_result,
        extended_metrics=extended_metrics_df,
        comparison=comparison_result,
    )

def main(argv: Optional[Sequence[str]] = None) -> Optional[PipelineResult]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="path to config file")
    parser.add_argument(
        "--stage",
        choices=list(STAGE_ORDER),
        nargs="+",
        help="limit execution to selected pipeline stages",
    )
    args = parser.parse_args(argv)
    config_paths = args.config
    stages = args.stage

    last_result: Optional[PipelineResult] = None
    for path in config_paths:
        parts = load_pipeline_config(path)
        pipeline_config = PipelineConfig(**parts)
        print(f"[tokenize_extension] Running config {path}")
        last_result = run_pipeline(pipeline_config, stages=stages)
    return last_result

if __name__ == "__main__":
    main()
