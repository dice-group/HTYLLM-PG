import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from lm_eval.utils import simple_parse_args_string
except Exception:  # pragma: no cover - fallback when lm_eval is unavailable
    def simple_parse_args_string(arg: str) -> dict:
        parsed: dict[str, str] = {}
        for item in arg.split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            key, value = item.split("=", 1)
            parsed[key.strip()] = value.strip()
        return parsed


def _load_entries(path: Path) -> list[dict]:
    entries: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def _init_run(run_id_dir: Path, suffix: str, wandb_args_dict: dict) -> "wandb.sdk.wandb_run.Run":
    run_id_path = run_id_dir / f".wandb_summary_id_{suffix}"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = f"sync_{suffix}_{int(time.time()*1e6)}"
        run_id_path.write_text(run_id)
    import wandb

    base_name = wandb_args_dict.get("name")
    if not base_name or base_name in {"eval", "summary"}:
        base_name = run_id_dir.parent.name or "eval"
    return wandb.init(
        id=run_id,
        resume="allow",
        name=f"{base_name}_{suffix}",
        settings=wandb.Settings(init_timeout=300),
        **wandb_args_dict,
    )


def _acquire_sync_lock(run_id_dir: Path, suffix: str) -> Path | None:
    lock_dir = run_id_dir / f".summary_sync_lock_{suffix}"
    ttl_s = int(os.environ.get("WANDB_SUMMARY_LOCK_TTL", "7200"))
    for _ in range(3):
        try:
            lock_dir.mkdir()
            return lock_dir
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > ttl_s:
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue
            print(
                f"[WARN] Summary sync lock held: {lock_dir} age_s={age:.0f}",
                file=sys.stderr,
            )
            return None
    return None


def _release_sync_lock(lock_dir: Path | None) -> None:
    if lock_dir is None:
        return
    try:
        lock_dir.rmdir()
    except OSError:
        pass


def _sync_series(series_path: Path, wandb_args: str, wandb_config_args: str | None) -> None:
    suffix = series_path.stem.replace("summary_series_", "")
    run_id_dir = series_path.parent
    entries = _load_entries(series_path)
    if not entries:
        print(f"[WARN] No entries found in {series_path}", file=sys.stderr)
        return
    entries = sorted(entries, key=lambda e: (e.get("step") is None, e.get("step", 0)))

    wandb_args_dict = simple_parse_args_string(wandb_args)
    project = wandb_args_dict.get("project")
    summary_suffix = os.environ.get("WANDB_SUMMARY_SUFFIX", "").strip()
    if project and summary_suffix:
        wandb_args_dict["project"] = f"{project}{summary_suffix}"
    wandb_args_dict.pop("resume", None)
    wandb_args_dict.pop("id", None)
    wandb_args_dict.pop("name", None)

    lock_dir = _acquire_sync_lock(run_id_dir, suffix)
    if lock_dir is None:
        return
    try:
        run = _init_run(run_id_dir, suffix, wandb_args_dict)
        print(
            f"[INFO] Summary sync run name={run.name} id={run.id} url={run.get_url()}",
            file=sys.stderr,
        )
        if wandb_config_args:
            cfg = simple_parse_args_string(wandb_config_args)
            if cfg:
                run.config.update(cfg, allow_val_change=True)
        for entry in entries:
            step = entry.get("step")
            metrics = entry.get("metrics", {})
            if step is None:
                run.log(metrics)
            else:
                run.log(metrics, step=step)
        run.finish()
    finally:
        _release_sync_lock(lock_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to lm_eval run dir")
    parser.add_argument("--wandb-args", required=True)
    parser.add_argument("--wandb-config-args", default=None)
    parser.add_argument("--suffix", default=None, help="Optional suffix filter (plain/no_ids/with_ids)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    pattern = "summary_series_*.jsonl" if args.suffix is None else f"summary_series_{args.suffix}.jsonl"
    series_paths = sorted(run_dir.glob(pattern))
    if not series_paths:
        raise SystemExit(f"No summary series files found under {run_dir}")
    for series_path in series_paths:
        _sync_series(series_path, args.wandb_args, args.wandb_config_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
