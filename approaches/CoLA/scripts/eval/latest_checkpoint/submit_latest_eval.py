import argparse
import os
import re
import shlex
import subprocess

from pathlib import Path
from typing import Iterable, Optional, Tuple


def _load_paths(paths: list[str], paths_file: Optional[str]) -> list[Path]:
    items: list[str] = []
    items.extend([p for p in paths if p])
    if paths_file:
        path_obj = Path(paths_file)
        if not path_obj.exists():
            raise SystemExit(f"[ERROR] paths file not found: {paths_file}")
        for line in path_obj.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                items.append(line)
    resolved: list[Path] = []
    for item in items:
        p = Path(item).expanduser().resolve()
        if not p.exists():
            print(f"[WARN] path not found, skipping: {p}")
            continue
        resolved.append(p)
    if not resolved:
        raise SystemExit("[ERROR] no valid paths provided")
    return resolved


def _is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("checkpoint-")


def _adapter_dir_for_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if (ckpt_dir / "adapter_config.json").exists():
        return ckpt_dir
    alt = Path(f"{ckpt_dir}_adapter")
    if (alt / "adapter_config.json").exists():
        return alt
    return None


def _checkpoint_step(path: Path) -> Optional[int]:
    match = re.search(r"checkpoint-(\d+)", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _find_latest_adapter_checkpoint(run_dir: Path) -> Optional[Tuple[Path, str]]:
    ckpts = [p for p in run_dir.iterdir() if _is_checkpoint_dir(p)]
    if not ckpts:
        return None
    candidates: list[Tuple[int, float, Path, str]] = []
    for ckpt in ckpts:
        adapter_dir = _adapter_dir_for_checkpoint(ckpt)
        if adapter_dir is None:
            continue
        step = _checkpoint_step(ckpt)
        mtime = ckpt.stat().st_mtime
        step_key = step if step is not None else -1
        candidates.append((step_key, mtime, adapter_dir, ckpt.name))
    if not candidates:
        return None
    # Prefer highest step; fall back to mtime.
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, adapter_dir, ckpt_name = candidates[-1]
    return adapter_dir, ckpt_name


def _label_from_run_dir(run_dir: Path) -> str:
    try:
        run_label = run_dir.name
        approach_label = run_dir.parent.name
        dataset_label = run_dir.parent.parent.name
        return f"{dataset_label}-{approach_label}-{run_label}"
    except Exception:
        return run_dir.name


def _build_sbatch_cmd(
    eval_script: Path,
    log_path: Path,
    env: dict[str, str],
    sbatch_args: list[str],
) -> list[str]:
    cmd = ["env", *[f"{k}={v}" for k, v in env.items()], "sbatch", f"--output={log_path}"]
    cmd.extend(sbatch_args)
    cmd.append(str(eval_script))
    return cmd


def _resolve_tasks_file(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (repo_root / path).resolve()
    return candidate


def _detect_lang_tier(path: Path) -> str | None:
    path_str = str(path)
    for tier in ("10_langs", "96_langs", "200_langs"):
        if tier in path_str:
            return tier
    return None


def _select_tasks_file(run_dir: Path, repo_root: Path, default_tasks: Path) -> Path:
    tier = _detect_lang_tier(run_dir)
    if tier is None:
        return default_tasks
    candidate_with_flores = repo_root / "configs" / f"lm_eval_tasks_{tier}_with_flores.txt"
    if candidate_with_flores.exists():
        return candidate_with_flores
    candidate = repo_root / "configs" / f"lm_eval_tasks_{tier}.txt"
    return candidate if candidate.exists() else default_tasks


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit eval jobs for latest adapter checkpoints in each run dir.")
    parser.add_argument("--paths", nargs="*", default=[], help="Run directories containing checkpoint-* folders.")
    parser.add_argument("--paths-file", default=None, help="File with one run path per line.")
    parser.add_argument("--eval-script", default="scripts/eval/latest_checkpoint/lm_eval_harness_latest.sh", help="SBATCH eval script to run.")
    parser.add_argument("--tasks-file", default="configs/lm_eval_tasks_200_langs_with_flores.txt", help="Tasks file path (default: configs/lm_eval_tasks_200_langs_with_flores.txt).")
    parser.add_argument("--auto-tasks-by-lang-tier", action="store_true", help="Auto-select lm_eval_tasks_{10,96,200}_langs.txt based on run path.")
    parser.add_argument("--wandb-project", default="htyllm-adapter-lpr-200_lang_cola_eval", help="W&B project name.")
    parser.add_argument("--wandb-entity", default="", help="W&B entity (optional).")
    parser.add_argument("--eval-partition", default="gpu", help="SBATCH partition for eval.")
    parser.add_argument("--eval-time", default="12:00:00", help="SBATCH time for eval.")
    parser.add_argument("--eval-gpus", type=int, default=1, help="GPUs for eval job.")
    parser.add_argument("--eval-gpu-type", default="h100", help="GPU type for eval job.")
    parser.add_argument("--eval-extra-sbatch", default="", help="Extra sbatch args.")
    parser.add_argument("--log-root", default="scripts/eval/logs/kiss", help="Directory for eval logs.")
    parser.add_argument("--output-subdir", default="lm_eval_latest", help="Subdir for eval outputs within run dir.")
    parser.add_argument("--wandb-name-suffix", default="", help="Optional suffix appended to WANDB_NAME.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output dir already has results.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    eval_script = Path(args.eval_script)
    if not eval_script.is_absolute():
        eval_script = (repo_root / args.eval_script).resolve()
    if not eval_script.exists():
        raise SystemExit(f"[ERROR] eval script not found: {eval_script}")

    default_tasks_file = _resolve_tasks_file(args.tasks_file, repo_root)
    if not default_tasks_file.exists():
        raise SystemExit(f"[ERROR] tasks file not found: {default_tasks_file}")

    run_paths = _load_paths(args.paths, args.paths_file)

    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = (repo_root / args.log_root).resolve()
    _ensure_dir(log_root)

    sbatch_args: list[str] = []
    if args.eval_partition:
        sbatch_args.append(f"--partition={args.eval_partition}")
    if args.eval_time:
        sbatch_args.append(f"--time={args.eval_time}")
    if args.eval_gpus and args.eval_gpus > 0:
        if args.eval_gpu_type:
            sbatch_args.append(f"--gres=gpu:{args.eval_gpu_type}:{args.eval_gpus}")
        else:
            sbatch_args.append(f"--gres=gpu:{args.eval_gpus}")
    if args.eval_extra_sbatch:
        sbatch_args.extend(shlex.split(args.eval_extra_sbatch))

    submitted = 0
    for run_dir in run_paths:
        latest = _find_latest_adapter_checkpoint(run_dir)
        if latest is None:
            print(f"[WARN] No adapter checkpoints found in {run_dir}")
            continue
        adapter_dir, ckpt_name = latest
        base_ckpt = ckpt_name.replace("_adapter", "")
        output_dir = run_dir / args.output_subdir / base_ckpt
        tasks_file = (
            _select_tasks_file(run_dir, repo_root, default_tasks_file)
            if args.auto_tasks_by_lang_tier
            else default_tasks_file
        )
        if not tasks_file.exists():
            raise SystemExit(f"[ERROR] tasks file not found: {tasks_file}")
        if args.skip_existing and output_dir.exists():
            # Skip if any result files exist.
            if any(output_dir.glob("*.json")) or any(output_dir.glob("*.jsonl")):
                print(f"[INFO] Skipping existing results: {output_dir}")
                continue

        label = _label_from_run_dir(run_dir)
        log_path = log_root / f"kiss_eval_{label}_{base_ckpt}.log"

        wandb_name_suffix = args.wandb_name_suffix.strip()
        wandb_name = f"{label}-{base_ckpt}"
        if wandb_name_suffix:
            wandb_name = f"{wandb_name}_{wandb_name_suffix}"

        env = {
            "CHECKPOINT_PATH": str(adapter_dir),
            "OUTPUT_DIR": str(output_dir),
            "TASKS_FILE": str(tasks_file),
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_GROUP": label,
            "WANDB_NAME": wandb_name,
            "REPO_ROOT": str(repo_root),
        }

        cmd = _build_sbatch_cmd(eval_script, log_path, env, sbatch_args)
        if args.dry_run:
            print("[DRY-RUN]", " ".join(shlex.quote(c) for c in cmd))
            continue
        subprocess.run(cmd, check=True)
        submitted += 1
        print(f"[INFO] Submitted eval for {label} @ {base_ckpt} tasks={tasks_file.name}")

    if submitted == 0:
        print("[WARN] No eval jobs submitted.")


if __name__ == "__main__":
    main()
