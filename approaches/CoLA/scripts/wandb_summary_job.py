import argparse
import os
import json
import sys
import time
from pathlib import Path

from lm_eval.utils import simple_parse_args_string


def _infer_checkpoint_step_from_paths(*paths: Path) -> int | None:
    import re

    for path in paths:
        if path is None:
            continue
        match = re.search(r"checkpoint-(\d+)", str(path))
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
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


def _parse_json_any(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return [parsed] if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        pass
    results: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
    if results:
        return results
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        try:
            parsed, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, dict):
            results.append(parsed)
        idx = end
        while idx < length and text[idx].isspace():
            idx += 1
    return results


def _load_latest_results_from_dir(path: Path) -> list[dict]:
    candidates = sorted(path.glob("results_*.json"))
    if not candidates:
        candidates = sorted(path.rglob("results_*.json"))
    if not candidates:
        print(
            f"[WARN] Summary job results dir empty: {path} (entries={len(list(path.iterdir()))})",
            file=sys.stderr,
        )
        return []
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        print(
            f"[INFO] Summary job loading results file: {latest}",
            file=sys.stderr,
        )
        return [json.loads(latest.read_text())]
    except Exception:
        print(
            f"[WARN] Summary job failed to parse results file: {latest}",
            file=sys.stderr,
        )
        return []


def _append_local_summary(
    run_id_dir: Path,
    suffix: str,
    step: int | None,
    checkpoint_path: Path,
    output_dir: Path,
    metrics: dict[str, float],
) -> None:
    lock_dir = run_id_dir / f".summary_lock_{suffix}"
    for _ in range(30):
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            time.sleep(0.2)
    else:
        print(
            f"[WARN] Summary job could not acquire lock for {suffix}; skipping local write",
            file=sys.stderr,
        )
        return
    try:
        series_path = run_id_dir / f"summary_series_{suffix}.jsonl"
        entry = {
            "suffix": suffix,
            "step": step,
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "metrics": metrics,
            "ts": time.time(),
        }
        with series_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(
            f"[INFO] Summary job wrote local summary: {series_path}",
            file=sys.stderr,
        )
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def _log_summary_series_wandb(
    output_dir: Path,
    checkpoint_path: Path,
    wandb_args: str,
    wandb_config_args: str | None,
) -> None:
    no_ids_path = output_dir / "no_language_ids.json"
    with_ids_paths = sorted(output_dir.glob("with_language_ids_*.json"))
    plain_paths = sorted(output_dir.glob("*_lm_eval.jsonl"))
    if not no_ids_path.exists() and not with_ids_paths and not plain_paths:
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
    if plain_paths:
        print(f"[INFO] Summary job plain_files={len(plain_paths)}", file=sys.stderr)
    if with_ids_paths:
        print(
            f"[INFO] Summary job with_ids_list={[p.name for p in with_ids_paths]}",
            file=sys.stderr,
        )
    if plain_paths:
        print(
            f"[INFO] Summary job plain_list={[p.name for p in plain_paths]}",
            file=sys.stderr,
        )

    no_ids_results = json.loads(no_ids_path.read_text()) if no_ids_path.exists() else {}
    with_ids_results = [json.loads(path.read_text()) for path in with_ids_paths]
    plain_results: list[dict] = []
    for path in plain_paths:
        if path.is_dir():
            parsed_list = _load_latest_results_from_dir(path)
            if parsed_list:
                print(
                    f"[INFO] Summary job using latest results in dir {path.name}",
                    file=sys.stderr,
                )
                plain_results.extend(parsed_list)
            else:
                print(
                    f"[WARN] Summary job found dir {path} but no parseable results_*.json",
                    file=sys.stderr,
                )
            continue
        try:
            text = path.read_text().strip()
        except Exception:
            continue
        if not text:
            continue
        parsed_list = _parse_json_any(text)
        if parsed_list:
            plain_results.extend(parsed_list)
        else:
            sample = "\\n".join(text.splitlines()[:3])
            print(
                f"[WARN] Summary job failed to parse {path.name} (size={path.stat().st_size}). "
                f"Head:\\n{sample}",
                file=sys.stderr,
            )
    if plain_paths and not plain_results:
        print(
            "[WARN] Summary job found plain files but no parseable JSON results.",
            file=sys.stderr,
        )
    summary_metrics = _summarize_results(with_ids_results or ([no_ids_results] if no_ids_results else []) or plain_results)
    step = _infer_checkpoint_step_from_paths(
        checkpoint_path,
        output_dir,
        output_dir.parent,
    )
    if step is None:
        print(
            f"[WARN] Summary job could not infer checkpoint step from {checkpoint_path}",
            file=sys.stderr,
        )
    else:
        print(f"[INFO] Summary job inferred step={step}", file=sys.stderr)

    import wandb

    wandb_args_dict = simple_parse_args_string(wandb_args)
    project = wandb_args_dict.get("project")
    summary_suffix = os.environ.get("WANDB_SUMMARY_SUFFIX", "").strip()
    if project and summary_suffix:
        wandb_args_dict["project"] = f"{project}{summary_suffix}"
    base_name = wandb_args_dict.get("name") or "eval"
    print(
        f"[INFO] Summary job run base_name={base_name} project={wandb_args_dict.get('project')}",
        file=sys.stderr,
    )
    wandb_args_dict.pop("resume", None)
    wandb_args_dict.pop("id", None)
    wandb_args_dict.pop("name", None)

    upload_mode = os.environ.get("WANDB_SUMMARY_UPLOAD", "true").strip().lower()
    upload_enabled = upload_mode not in {"0", "false", "no", "local", "offline"}
    if not upload_enabled:
        print(
            f"[INFO] Summary job running local-only (WANDB_SUMMARY_UPLOAD={upload_mode})",
            file=sys.stderr,
        )

    def init_series_run(suffix: str) -> "wandb.sdk.wandb_run.Run":
        run_id_path = run_id_dir / f".wandb_summary_id_{suffix}"
        if run_id_path.exists():
            run_id = run_id_path.read_text().strip()
        else:
            run_id = f"summary_{suffix}_{int(time.time()*1e6)}"
            run_id_path.write_text(run_id)
        run = wandb.init(
            id=run_id,
            resume="allow",
            name=f"{base_name}_{suffix}",
            settings=wandb.Settings(init_timeout=300),
            **wandb_args_dict,
        )
        print(
            f"[INFO] Summary job wandb_run name={run.name} id={run.id} url={run.get_url()}",
            file=sys.stderr,
        )
        return run

    if no_ids_results:
        run = init_series_run("no_ids") if upload_enabled else None
        if run is not None:
            print(f"[INFO] Summary job W&B url={run.get_url()}", file=sys.stderr)
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
        if step is not None:
            metrics["summary/checkpoint_step"] = float(step)
        _append_local_summary(run_id_dir, "no_ids", step, checkpoint_path, output_dir, metrics)
        if run is not None:
            if step is None:
                run.log(metrics)
            else:
                run.log(metrics, step=step)
            run.finish()

    if with_ids_results:
        run = init_series_run("with_ids") if upload_enabled else None
        if run is not None:
            print(f"[INFO] Summary job W&B url={run.get_url()}", file=sys.stderr)
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
        if step is not None:
            metrics["summary/checkpoint_step"] = float(step)
        _append_local_summary(run_id_dir, "with_ids", step, checkpoint_path, output_dir, metrics)
        if run is not None:
            if step is None:
                run.log(metrics)
            else:
                run.log(metrics, step=step)
            run.finish()

    if plain_results:
        run = init_series_run("plain") if upload_enabled else None
        if run is not None and wandb_config_args:
            cfg = simple_parse_args_string(wandb_config_args)
            if cfg:
                run.config.update(cfg, allow_val_change=True)
        metrics: dict[str, float] = {}
        for entry in plain_results:
            for task_name, vals in entry.get("results", {}).items():
                if not isinstance(vals, dict):
                    continue
                for key, value in vals.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{task_name}/{key}"] = float(value)
        for key, value in summary_metrics.items():
            metrics[f"summary/{key}"] = float(value)
        if step is not None:
            metrics["summary/checkpoint_step"] = float(step)
        _append_local_summary(run_id_dir, "plain", step, checkpoint_path, output_dir, metrics)
        if run is not None:
            if step is None:
                run.log(metrics)
            else:
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
