import argparse
import json
import sys
import time
from pathlib import Path

from lm_eval.utils import simple_parse_args_string


def _infer_checkpoint_step(checkpoint_path: Path) -> int | None:
    import re

    match = re.search(r"checkpoint-(\d+)", str(checkpoint_path))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _summarize_results(results_list: list[dict]) -> dict[str, float]:
    if not results_list:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for results in results_list:
        task_results = results.get("results", {})
        for task_name, metrics in task_results.items():
            if task_name.startswith("_"):
                continue
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k)}


def _log_summary_series_wandb(
    output_dir: Path,
    checkpoint_path: Path,
    wandb_args: str,
    wandb_config_args: str | None,
) -> None:
    no_ids_path = output_dir / "no_language_ids.json"
    with_ids_paths = sorted(output_dir.glob("with_language_ids_*.json"))
    if not no_ids_path.exists() and not with_ids_paths:
        print(f"[WARN] No eval results found under {output_dir}", file=sys.stderr)
        return
    if "checkpoint-" not in output_dir.name:
        print(
            f"[WARN] Summary job expects per-checkpoint output dir, got {output_dir}",
            file=sys.stderr,
        )
        return
    run_id_dir = output_dir.parent
    print(f"[INFO] Summary job output_dir={output_dir}", file=sys.stderr)
    print(f"[INFO] Summary job run_id_dir={run_id_dir}", file=sys.stderr)
    print(f"[INFO] Summary job checkpoint={checkpoint_path}", file=sys.stderr)
    if no_ids_path.exists():
        print(f"[INFO] Summary job found {no_ids_path.name}", file=sys.stderr)
    print(f"[INFO] Summary job with_ids_files={len(with_ids_paths)}", file=sys.stderr)

    no_ids_results = json.loads(no_ids_path.read_text()) if no_ids_path.exists() else {}
    with_ids_results = [json.loads(path.read_text()) for path in with_ids_paths]
    summary_metrics = _summarize_results(with_ids_results or [no_ids_results])
    step = _infer_checkpoint_step(checkpoint_path)

    import wandb

    wandb_args_dict = simple_parse_args_string(wandb_args)
    project = wandb_args_dict.get("project")
    if project:
        wandb_args_dict["project"] = f"{project}_summary"
    base_name = wandb_args_dict.get("name") or "eval"
    print(f"[INFO] Summary job run base_name={base_name}", file=sys.stderr)
    wandb_args_dict.pop("resume", None)
    wandb_args_dict.pop("id", None)
    wandb_args_dict.pop("name", None)

    def init_series_run(suffix: str) -> "wandb.sdk.wandb_run.Run":
        run_id_path = run_id_dir / f".wandb_summary_id_{suffix}"
        if run_id_path.exists():
            run_id = run_id_path.read_text().strip()
        else:
            run_id = f"summary_{suffix}_{int(time.time()*1e6)}"
            run_id_path.write_text(run_id)
        return wandb.init(
            id=run_id,
            resume="allow",
            name=f"{base_name}_{suffix}",
            settings=wandb.Settings(init_timeout=300),
            **wandb_args_dict,
        )

    if no_ids_results:
        run = init_series_run("no_ids")
        if wandb_config_args:
            cfg = simple_parse_args_string(wandb_config_args)
            if cfg:
                run.config.update(cfg, allow_val_change=True)
        metrics: dict[str, float] = {}
        for task_name, vals in no_ids_results.get("results", {}).items():
            if not isinstance(vals, dict):
                continue
            for key, value in vals.items():
                if isinstance(value, (int, float)):
                    metrics[f"{task_name}/{key}"] = float(value)
        for key, value in summary_metrics.items():
            metrics[f"summary/{key}"] = float(value)
        run.log(metrics, step=step)
        run.finish()

    if with_ids_results:
        run = init_series_run("with_ids")
        if wandb_config_args:
            cfg = simple_parse_args_string(wandb_config_args)
            if cfg:
                run.config.update(cfg, allow_val_change=True)
        metrics = {}
        for entry in with_ids_results:
            for task_name, vals in entry.get("results", {}).items():
                if not isinstance(vals, dict):
                    continue
                for key, value in vals.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{task_name}/{key}"] = float(value)
        for key, value in summary_metrics.items():
            metrics[f"summary/{key}"] = float(value)
        run.log(metrics, step=step)
        run.finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wandb-args", required=True)
    parser.add_argument("--wandb-config-args", default=None)
    args = parser.parse_args()

    _log_summary_series_wandb(
        Path(args.output_dir),
        Path(args.checkpoint),
        args.wandb_args,
        args.wandb_config_args,
    )
    print("[INFO] Summary job completed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
