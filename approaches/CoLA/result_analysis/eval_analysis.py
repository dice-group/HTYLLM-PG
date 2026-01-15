import argparse
import json
import re
import numpy as np
import pandas as pd
import wandb

import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

"""
Aggregate lm-eval results from W&B, compute paper-ready summaries, and emit plots.

Defaults are tuned for the tier200 multilingual ablation setup, but all projects
and groups are configurable via CLI.
"""

DEFAULT_TRAIN_PROJECT = "dice-nlp/htyllm-adapter-lpr-200_lang_cola"
DEFAULT_EVAL_PROJECT = "dice-nlp/htyllm-adapter-lpr-200_lang_cola_eval"
DEFAULT_TASK_PREFIX = "belebele_"

WITH_IDS_RE = re.compile(r"_with_ids_(.+)_detailed")
NO_IDS_RE = re.compile(r"_no_ids_detailed")
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


COLA_ROUTER_KEYS = {
    "cola_expert_load_cv": "train/train/cola/expert_load_cv",
    "cola_active_expert_frac": "train/train/cola/active_expert_frac",
    "cola_router_entropy": "train/train/cola/router_entropy",
    "cola_topk_weight_mean": "train/train/cola/topk_weight_mean",
    "cola_language_target_hit_rate": "train/train/cola/language_target_hit_rate",
    "cola_language_target_prob_mean": "train/train/cola/language_target_prob_mean",
    "cola_language_target_neglogp": "train/train/cola/language_target_neglogp",
    "cola_language_target_token_frac": "train/train/cola/language_target_token_frac",
    "cola_language_prior_loss": "train/train/cola/language_prior_loss",
}

HYDRA_ROUTER_KEYS = {
    "hydra_expert_load_cv": "train/train/hydralora/expert_load_cv",
    "hydra_expert_active_frac": "train/train/hydralora/expert_active_frac",
    "hydra_expert_router_entropy": "train/train/hydralora/expert_router_entropy",
    "hydra_expert_topk_weight_mean": "train/train/hydralora/expert_topk_weight_mean",
    "hydra_expert_target_hit_rate": "train/train/hydralora/expert_target_hit_rate",
    "hydra_expert_target_prob_mean": "train/train/hydralora/expert_target_prob_mean",
    "hydra_expert_target_neglogp": "train/train/hydralora/expert_target_neglogp",
    "hydra_expert_target_token_frac": "train/train/hydralora/expert_target_token_frac",
}


@dataclass
class TrainRunInfo:
    run_id: str
    name: str
    group: Optional[str]
    adapter: Optional[str]
    variant: Optional[str]
    mode: Optional[str]
    head_mode: Optional[str]
    gamma: Optional[str]
    tier: Optional[str]
    summary: dict


def _parse_tags(tags: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for tag in tags or []:
        if ":" in tag:
            key, value = tag.split(":", 1)
            parsed[key] = value
    return parsed


def _load_resource_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = pd.read_csv(path, sep="\t")
    mapping = {}
    for _, row in data.iterrows():
        lang = str(row["lang_code"]).lower()
        mapping[lang] = str(row.get("resource_category", "")).strip()
    return mapping


def _task_from_run_name(name: str) -> Optional[str]:
    match = WITH_IDS_RE.search(name)
    if match:
        return match.group(1)
    return None


def _tasks_from_summary(summary: dict, prefix: str) -> set[str]:
    tasks = set()
    for key in summary.keys():
        if not key.startswith(prefix):
            continue
        if key.endswith("/acc") or key.endswith("/acc_norm"):
            task = key.split("/")[0]
            tasks.add(task)
    return tasks


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _metric_for_task(summary: dict, task: str) -> Optional[float]:
    value = summary.get(f"{task}/acc_norm")
    if value is None:
        value = summary.get(f"{task}/acc")
    return _coerce_float(value)


def _extract_checkpoint(name: str) -> Optional[int]:
    match = CHECKPOINT_RE.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _collect_train_runs(api: wandb.Api, project: str) -> list[TrainRunInfo]:
    runs = api.runs(project)
    result: list[TrainRunInfo] = []
    for run in runs:
        tags = _parse_tags(run.tags or [])
        output_dir = run.config.get("output_dir")
        group = Path(output_dir).name if output_dir else None
        result.append(
            TrainRunInfo(
                run_id=run.id,
                name=run.name or run.id,
                group=group,
                adapter=tags.get("adapter"),
                variant=tags.get("variant"),
                mode=tags.get("mode"),
                head_mode=tags.get("head_mode"),
                gamma=tags.get("gamma"),
                tier=tags.get("tier"),
                summary=run.summary,
            )
        )
    return result


def _eval_runs_for_group(
    api: wandb.Api,
    project: str,
    group: str,
    mode: str,
    task_prefix: str,
) -> list[wandb.apis.public.Run]:
    name_regex = "with_ids" if mode == "with_ids" else "no_ids"
    if mode == "any":
        name_regex = ""
    filters = {"group": group}
    if name_regex:
        filters["display_name"] = {"$regex": name_regex}
    runs = list(api.runs(project, filters=filters))
    if not runs:
        return []
    if mode != "any":
        return runs
    # For any-mode, drop router-only tables without eval tasks.
    return [
        run
        for run in runs
        if _tasks_from_summary(run.summary, task_prefix)
        or _task_from_run_name(run.name or "")
    ]


def _aggregate_eval(
    api: wandb.Api,
    project: str,
    group: str,
    mode: str,
    task_prefix: str,
) -> tuple[int, dict[str, float]]:
    runs = _eval_runs_for_group(api, project, group, mode, task_prefix)
    if not runs:
        return 0, {}
    checkpoints = [
        ckpt
        for run in runs
        if (ckpt := _extract_checkpoint(run.name or "")) is not None
    ]
    if not checkpoints:
        return 0, {}
    target_ckpt = max(checkpoints)
    task_scores: dict[str, float] = {}
    for run in runs:
        name = run.name or ""
        if f"checkpoint-{target_ckpt}" not in name:
            continue
        task = _task_from_run_name(name)
        if task is None:
            candidates = _tasks_from_summary(run.summary, task_prefix)
            if len(candidates) == 1:
                task = next(iter(candidates))
        if task is None or not task.startswith(task_prefix):
            continue
        if task in task_scores:
            continue
        score = _metric_for_task(run.summary, task)
        if score is None:
            continue
        task_scores[task] = score
    return target_ckpt, task_scores


def _summarize_task_scores(
    task_scores: dict[str, float],
    resource_map: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if not task_scores:
        return {}, {}, {}
    values = list(task_scores.values())
    overall = {
        "acc_mean": float(np.mean(values)),
        "acc_median": float(np.median(values)),
        "acc_std": float(np.std(values)),
    }
    resource_buckets: dict[str, list[float]] = defaultdict(list)
    script_buckets: dict[str, list[float]] = defaultdict(list)
    for task, value in task_scores.items():
        lang = task.split("_", 1)[1] if "_" in task else task
        resource = resource_map.get(lang.lower(), "unknown")
        resource_buckets[resource].append(value)
        parts = lang.split("_")
        script = parts[1] if len(parts) > 1 else "unknown"
        script_buckets[script].append(value)
    resource_means = {k: float(np.mean(v)) for k, v in resource_buckets.items() if v}
    script_means = {k: float(np.mean(v)) for k, v in script_buckets.items() if v}
    return overall, resource_means, script_means


def _build_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _plot_overall(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    if plt is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = df["label"].tolist()
    ax.bar(labels, df["acc_mean"])
    ax.set_ylabel("Belebele acc_norm (macro)")
    ax.set_title("Macro accuracy by variant")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_dir / f"overall_accuracy.{fmt}")
    plt.close(fig)


def _plot_resource(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    if plt is None or df.empty:
        return
    resource_cols = [col for col in df.columns if col.startswith("resource/")]
    if not resource_cols:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    width = 0.8 / max(len(resource_cols), 1)
    for idx, col in enumerate(sorted(resource_cols)):
        values = df[col].fillna(0)
        ax.bar(x + idx * width, values, width, label=col.replace("resource/", ""))
    ax.set_xticks(x + width * (len(resource_cols) - 1) / 2)
    ax.set_xticklabels(df["label"].tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Belebele acc_norm (macro)")
    ax.set_title("Accuracy by resource tier")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_dir / f"resource_accuracy.{fmt}")
    plt.close(fig)


def _plot_router_scatter(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    if plt is None or df.empty:
        return
    metrics = [col for col in df.columns if col.startswith("router/")]
    if not metrics:
        return
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df[metric], df["acc_mean"])
        for _, row in df.iterrows():
            ax.annotate(row["label"], (row[metric], row["acc_mean"]), fontsize=7, alpha=0.7)
        ax.set_xlabel(metric.replace("router/", ""))
        ax.set_ylabel("Belebele acc_norm (macro)")
        ax.set_title("Router metric vs accuracy")
        fig.tight_layout()
        safe_metric = metric.replace("/", "_")
        for fmt in formats:
            fig.savefig(output_dir / f"router_vs_{safe_metric}.{fmt}")
        plt.close(fig)


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [col for col in df.columns if col.startswith("router/")]
    rows = []
    for col in cols:
        series = df[[col, "acc_mean"]].dropna()
        if len(series) < 3:
            continue
        corr = float(series[col].corr(series["acc_mean"]))
        rows.append({"metric": col, "pearson_r": corr, "n": len(series)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-project", default=DEFAULT_TRAIN_PROJECT)
    parser.add_argument("--eval-project", default=DEFAULT_EVAL_PROJECT)
    parser.add_argument("--mode", choices=["with_ids", "no_ids", "any"], default="with_ids")
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--resource-map", default="data_prep/base_data/lang_resource_dataset.tsv")
    parser.add_argument("--output-dir", default="result_analysis/paper_eval")
    parser.add_argument("--plot-formats", default="png")
    parser.add_argument("--api-timeout", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip() for fmt in args.plot_formats.split(",") if fmt.strip()]

    api = wandb.Api(timeout=args.api_timeout)
    resource_map = _load_resource_map(Path(args.resource_map))

    train_runs = _collect_train_runs(api, args.train_project)
    per_run_rows = []
    per_task_rows = []

    for info in train_runs:
        if not info.group:
            continue
        ckpt, task_scores = _aggregate_eval(
            api,
            args.eval_project,
            info.group,
            args.mode,
            args.task_prefix,
        )
        overall, resource_means, script_means = _summarize_task_scores(
            task_scores, resource_map
        )
        label = info.variant or info.name
        row = {
            "label": label,
            "adapter": info.adapter,
            "variant": info.variant,
            "mode": info.mode,
            "head_mode": info.head_mode,
            "gamma": info.gamma,
            "tier": info.tier,
            "eval_group": info.group,
            "eval_ckpt": ckpt,
            "task_count": len(task_scores),
            "train_loss": _coerce_float(info.summary.get("train/loss")),
            "eval_loss": _coerce_float(info.summary.get("eval/loss")),
        }
        row.update(overall)
        for key, value in resource_means.items():
            row[f"resource/{key}"] = value
        for key, value in script_means.items():
            row[f"script/{key}"] = value
        for metric_key, summary_key in COLA_ROUTER_KEYS.items():
            value = _coerce_float(info.summary.get(summary_key))
            if value is not None:
                row[f"router/{metric_key}"] = value
        for metric_key, summary_key in HYDRA_ROUTER_KEYS.items():
            value = _coerce_float(info.summary.get(summary_key))
            if value is not None:
                row[f"router/{metric_key}"] = value
        per_run_rows.append(row)

        for task, score in task_scores.items():
            lang = task.split("_", 1)[1] if "_" in task else task
            per_task_rows.append(
                {
                    "label": label,
                    "adapter": info.adapter,
                    "variant": info.variant,
                    "eval_group": info.group,
                    "eval_ckpt": ckpt,
                    "task": task,
                    "language": lang,
                    "resource": resource_map.get(lang.lower(), "unknown"),
                    "script": lang.split("_")[1] if "_" in lang else "unknown",
                    "acc_norm": score,
                }
            )

    per_run_df = _build_dataframe(per_run_rows)
    per_task_df = _build_dataframe(per_task_rows)

    per_run_csv = output_dir / "per_run_summary.csv"
    per_task_csv = output_dir / "per_task_scores.csv"
    per_run_df.to_csv(per_run_csv, index=False)
    per_task_df.to_csv(per_task_csv, index=False)

    corr_df = _compute_correlations(per_run_df)
    corr_csv = output_dir / "correlations.csv"
    corr_df.to_csv(corr_csv, index=False)

    _plot_overall(per_run_df, output_dir, formats)
    _plot_resource(per_run_df, output_dir, formats)
    _plot_router_scatter(per_run_df, output_dir, formats)

    report = {
        "train_project": args.train_project,
        "eval_project": args.eval_project,
        "mode": args.mode,
        "task_prefix": args.task_prefix,
        "run_count": len(per_run_df),
        "task_count": len(per_task_df),
        "per_run_csv": str(per_run_csv),
        "per_task_csv": str(per_task_csv),
        "correlations_csv": str(corr_csv),
        "plots": formats,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
