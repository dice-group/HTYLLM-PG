#!/usr/bin/env python3
"""
Aggregate metrics for the Language-Prior ablation runs launched via
`scripts/comparison/run_language_prior_ablation.sh`.

For each run type (CoLA MoE experts, CoLA flat, HydraLoRA) and each label
(baseline / soft_prior / bias_prior / hard_prior), this script:
  * pulls the most recent run from the specified W&B project/group,
  * collects metric histories, computing first/mid/latest/min/max/mean,
  * records key configuration fields, and
  * writes everything to `result_analysis/lpr_metrics_snapshot.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import wandb


# Canonical metric names -> list of aliases to try when fetching history.
METRIC_ALIASES: Dict[str, List[str]] = {
    "train/loss": ["train/loss", "loss"],
    "train/learning_rate": ["train/learning_rate", "learning_rate"],
    "train/cola/expert_load_cv": ["train/cola/expert_load_cv", "cola/expert_load_cv"],
    "train/cola/active_expert_frac": ["train/cola/active_expert_frac", "cola/active_expert_frac"],
    "train/cola/router_entropy": ["train/cola/router_entropy", "cola/router_entropy"],
    "train/cola/topk_weight_mean": ["train/cola/topk_weight_mean", "cola/topk_weight_mean"],
    "train/cola/language_target_hit_rate": [
        "train/cola/language_target_hit_rate",
        "cola/language_target_hit_rate",
    ],
    "train/cola/language_target_prob_mean": [
        "train/cola/language_target_prob_mean",
        "cola/language_target_prob_mean",
    ],
    "train/cola/language_target_token_frac": [
        "train/cola/language_target_token_frac",
        "cola/language_target_token_frac",
    ],
    "train/hydralora/head_load_cv": [
        "train/hydralora/head_load_cv",
        "hydralora/head_load_cv",
    ],
    "train/hydralora/head_active_frac": [
        "train/hydralora/head_active_frac",
        "hydralora/head_active_frac",
    ],
    "train/hydralora/head_router_entropy": [
        "train/hydralora/head_router_entropy",
        "hydralora/head_router_entropy",
    ],
    "train/hydralora/head_target_hit_rate": [
        "train/hydralora/head_target_hit_rate",
        "hydralora/head_target_hit_rate",
    ],
    "train/hydralora/head_target_prob_mean": [
        "train/hydralora/head_target_prob_mean",
        "hydralora/head_target_prob_mean",
    ],
    "train/hydralora/head_target_token_frac": [
        "train/hydralora/head_target_token_frac",
        "hydralora/head_target_token_frac",
    ],
    "train/hydralora/expert_load_cv": [
        "train/hydralora/expert_load_cv",
        "hydralora/expert_load_cv",
    ],
    "train/hydralora/expert_active_frac": [
        "train/hydralora/expert_active_frac",
        "hydralora/expert_active_frac",
    ],
    "train/hydralora/expert_router_entropy": [
        "train/hydralora/expert_router_entropy",
        "hydralora/expert_router_entropy",
    ],
    "train/hydralora/expert_topk_weight_mean": [
        "train/hydralora/expert_topk_weight_mean",
        "hydralora/expert_topk_weight_mean",
    ],
    "train/hydralora/expert_target_hit_rate": [
        "train/hydralora/expert_target_hit_rate",
        "hydralora/expert_target_hit_rate",
    ],
    "train/hydralora/expert_target_prob_mean": [
        "train/hydralora/expert_target_prob_mean",
        "hydralora/expert_target_prob_mean",
    ],
    "train/hydralora/expert_target_token_frac": [
        "train/hydralora/expert_target_token_frac",
        "hydralora/expert_target_token_frac",
    ],
}


RUN_TYPES = {
    "cola_experts": {
        "prefix": "colaexp-",
        "required_metrics": [
            "train/loss",
            "train/cola/language_target_hit_rate",
        ],
        "optional_metrics": [
            "train/learning_rate",
            "train/cola/expert_load_cv",
            "train/cola/active_expert_frac",
            "train/cola/router_entropy",
            "train/cola/topk_weight_mean",
            "train/cola/language_target_prob_mean",
            "train/cola/language_target_token_frac",
        ],
    },
    "cola_flat": {
        "prefix": "colaflat-",
        "required_metrics": [
            "train/loss",
            "train/cola/language_target_hit_rate",
        ],
        "optional_metrics": [
            "train/learning_rate",
            "train/cola/language_target_prob_mean",
            "train/cola/language_target_token_frac",
        ],
    },
    "hydralora": {
        "prefix": "hydra-",
        "required_metrics": [
            "train/loss",
            "train/hydralora/head_target_hit_rate",
        ],
        "optional_metrics": [
            "train/learning_rate",
            "train/hydralora/head_load_cv",
            "train/hydralora/head_active_frac",
            "train/hydralora/head_router_entropy",
            "train/hydralora/head_target_prob_mean",
            "train/hydralora/head_target_token_frac",
            "train/hydralora/expert_load_cv",
            "train/hydralora/expert_active_frac",
            "train/hydralora/expert_router_entropy",
            "train/hydralora/expert_topk_weight_mean",
            "train/hydralora/expert_target_hit_rate",
            "train/hydralora/expert_target_prob_mean",
            "train/hydralora/expert_target_token_frac",
        ],
    },
}

# Config keys of interest to include in the snapshot.
CONFIG_FIELDS = [
    "language_router_mode",
    "language_prior_weight",
    "language_bias_value",
    "use_cola_experts",
    "use_hydralora_experts",
    "cola_num_experts",
    "cola_top_k",
    "num_A",
    "num_B",
    "lora_rank",
    "lora_num",
    "hydralora_num_experts",
    "hydralora_top_k",
]


@dataclass
class MetricStats:
    latest: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    first: Optional[float] = None
    mid: Optional[float] = None

    @classmethod
    def from_history(cls, values: List[float]) -> "MetricStats":
        if not values:
            return cls()
        mid_idx = len(values) // 2
        return cls(
            latest=values[-1],
            minimum=min(values),
            maximum=max(values),
            mean=statistics.fmean(values),
            first=values[0],
            mid=values[mid_idx],
        )

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "latest": self.latest,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "first": self.first,
            "mid": self.mid,
        }


@dataclass
class RunSnapshot:
    name: str
    run_id: str
    label: str
    url: str
    state: str
    missing_required: List[str]
    metrics: Dict[str, MetricStats] = field(default_factory=dict)
    config: Dict[str, Optional[float | str | int | bool]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "id": self.run_id,
            "label": self.label,
            "state": self.state,
            "url": self.url,
            "missing_required": self.missing_required,
            "config": self.config,
            "metrics": {metric: stats.to_dict() for metric, stats in self.metrics.items()},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Weights & Biases entity/organization.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "htyllm-adapter-lpr"),
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="Run group to filter on (e.g., lpr-ablation-20250101_120000).",
    )
    parser.add_argument(
        "--output",
        default="result_analysis/lpr_metrics_snapshot.json",
        help="Path to write the aggregated JSON payload.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=400,
        help="Maximum number of history points to pull per metric.",
    )
    return parser.parse_args()


def build_project_path(entity: Optional[str], project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    return project


def collect_metric_history(
    run: wandb.apis.public.Run,
    metric: str,
    aliases: Dict[str, List[str]],
    max_points: int,
) -> List[float]:
    names = aliases.get(metric, [metric])
    for name in names:
        try:
            history_rows = run.history(keys=[name], pandas=False)
        except wandb.errors.CommError:
            continue
        values = [
            row.get(name)
            for row in history_rows[:max_points]
            if name in row and isinstance(row.get(name), (int, float))
        ]
        if values:
            return values
    return []


def extract_label(run_name: str, prefix: str) -> str:
    remainder = run_name[len(prefix) :]
    if "-" not in remainder:
        return remainder
    return remainder.rsplit("-", 1)[0]


def summarize_run(
    run: wandb.apis.public.Run,
    run_type: str,
    label: str,
    definition: Dict[str, object],
    max_points: int,
) -> RunSnapshot:
    required = definition["required_metrics"]
    optional = definition["optional_metrics"]
    metrics = {}
    missing_required: List[str] = []

    for metric in required + optional:
        values = collect_metric_history(run, metric, METRIC_ALIASES, max_points=max_points)
        stats = MetricStats.from_history(values)
        metrics[metric] = stats
        if metric in required and stats.latest is None:
            missing_required.append(metric)

    config_snapshot = {
        key: run.config.get(key) for key in CONFIG_FIELDS if key in run.config
    }

    return RunSnapshot(
        name=run.name or run.id,
        run_id=run.id,
        label=label,
        url=run.url,
        state=run.state,
        missing_required=missing_required,
        metrics=metrics,
        config=config_snapshot,
    )


def gather_runs(project_path: str, group: str, max_points: int) -> Dict[str, Dict[str, RunSnapshot]]:
    api = wandb.Api()
    runs = api.runs(project_path, filters={"group": group})
    snapshots: Dict[str, Dict[str, RunSnapshot]] = {
        key: {} for key in RUN_TYPES
    }

    for run in runs:
        run_name = run.name or run.id
        for run_type, definition in RUN_TYPES.items():
            prefix = definition["prefix"]
            if not run_name.startswith(prefix):
                continue
            label = extract_label(run_name, prefix)
            # Keep most recent run per label/type (runs are already returned sorted newest-first).
            if label in snapshots[run_type]:
                continue
            snapshots[run_type][label] = summarize_run(
                run=run,
                run_type=run_type,
                label=label,
                definition=definition,
                max_points=max_points,
            )
            break

    return snapshots


def write_snapshot(
    data: Dict[str, Dict[str, RunSnapshot]],
    output_path: Path,
    project: str,
    group: str,
) -> None:
    payload = {
        "project": project,
        "group": group,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": {
            run_type: {
                label: snapshot.to_dict() for label, snapshot in sorted(runs.items())
            }
            for run_type, runs in data.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_path = build_project_path(args.entity, args.project)

    print(f"[INFO] Querying W&B project '{project_path}' group '{args.group}'")
    snapshots = gather_runs(project_path, args.group, max_points=args.max_points)
    missing_types = [run_type for run_type, runs in snapshots.items() if not runs]
    if missing_types:
        print(f"[WARN] No runs found for: {', '.join(missing_types)}")

    output_path = Path(args.output)
    write_snapshot(snapshots, output_path, project=project_path, group=args.group)
    print(f"[INFO] Wrote snapshot to {output_path}")

    for run_type, runs in snapshots.items():
        if not runs:
            continue
        print(f"\n=== {run_type.upper()} ===")
        for label, snapshot in sorted(runs.items()):
            badge = "[OK]" if not snapshot.missing_required and snapshot.state == "finished" else "[ISSUE]"
            print(f"{badge} {label}: {snapshot.name} ({snapshot.state}) -> {snapshot.url}")
            if snapshot.missing_required:
                print(f"    Missing required metrics: {', '.join(snapshot.missing_required)}")
            for metric, stats in snapshot.metrics.items():
                if stats.latest is None:
                    continue
                print(
                    f"    - {metric}: first={stats.first!r}, mid={stats.mid!r}, "
                    f"latest={stats.latest!r}, min={stats.minimum!r}, max={stats.maximum!r}, "
                    f"mean={stats.mean!r}"
                )


if __name__ == "__main__":
    try:
        main()
    except wandb.errors.CommError as exc:
        print(f"[ERROR] Failed to query Weights & Biases: {exc}", file=sys.stderr)
        sys.exit(1)
