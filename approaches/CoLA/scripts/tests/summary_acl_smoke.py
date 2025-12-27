from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_last_metric(log_history: list[dict], key: str) -> float | None:
    for entry in reversed(log_history):
        if key in entry:
            return entry[key]
    return None


def _find_last_prefix(log_history: list[dict], prefix: str) -> tuple[str, float] | None:
    for entry in reversed(log_history):
        for k, v in entry.items():
            if k.startswith(prefix):
                return k, v
    return None


def _find_last_any(log_history: list[dict], keys: list[str]) -> float | None:
    for key in keys:
        value = _find_last_metric(log_history, key)
        if value is not None:
            return value
    return None


def _find_total_flos(results: dict, log_history: list[dict]) -> float | None:
    total = results.get("total_flos")
    if total is not None:
        return total
    for entry in reversed(log_history):
        if "total_flos" in entry:
            return entry["total_flos"]
    return None


def _compute_tflops(total_flos: float | None, train_runtime: float | None) -> float | None:
    if total_flos is None or not train_runtime:
        return None
    return float(total_flos) / float(train_runtime) / 1e12


def _compute_device_flops(world_size: int) -> float | None:
    if not torch.cuda.is_available():
        return None
    return 990 * 1e12 * world_size


def _compute_mfu(tflops: float | None, world_size: int) -> float | None:
    if tflops is None:
        return None
    device_flops = _compute_device_flops(world_size)
    if device_flops is None:
        return None
    return (tflops * 1e12) / device_flops


def _format(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _collect_runs(root: Path) -> list[dict]:
    runs = []
    for output_dir in sorted(root.glob("*_*")):
        if not output_dir.is_dir():
            continue
        trainer_state = output_dir / "trainer_state.json"
        train_results = output_dir / "train_results.json"
        if not trainer_state.exists() or not train_results.exists():
            continue
        state = _load_json(trainer_state)
        results = _load_json(train_results)
        history = state.get("log_history", [])
        total_flos = _find_total_flos(results, history)
        tflops = _compute_tflops(total_flos, results.get("train_runtime"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        mfu = _compute_mfu(tflops, world_size)
        hit_rate = _find_last_any(
            history,
            [
                "train/cola/language_target_hit_rate",
                "cola/language_target_hit_rate",
                "train/hydralora/expert_target_hit_rate",
                "hydralora/expert_target_hit_rate",
                "train/hydralora/language_target_hit_rate",
                "hydralora/language_target_hit_rate",
            ],
        )
        load_cv = _find_last_any(
            history,
            [
                "train/cola/expert_load_cv",
                "cola/expert_load_cv",
                "train/hydralora/expert_load_cv",
                "hydralora/expert_load_cv",
            ],
        )
        active_frac = _find_last_any(
            history,
            [
                "train/cola/active_expert_frac",
                "cola/active_expert_frac",
                "train/hydralora/active_expert_frac",
                "hydralora/active_expert_frac",
            ],
        )
        entropy = _find_last_any(
            history,
            [
                "train/cola/router_entropy",
                "cola/router_entropy",
                "train/hydralora/router_entropy",
                "hydralora/router_entropy",
            ],
        )
        run = {
            "run": output_dir.name,
            "global_step": state.get("global_step", 0),
            "train_loss": results.get("train_loss"),
            "train_runtime": results.get("train_runtime"),
            "tps": results.get("effective_tokens_per_sec"),
            "tflops": tflops,
            "mfu": mfu,
            "lpr_loss": _find_last_metric(history, "language_prior_loss"),
            "hit_rate": hit_rate,
            "load_cv": load_cv,
            "active_frac": active_frac,
            "entropy": entropy,
        }
        runs.append(run)
    return runs


def _print_table(rows: list[dict]) -> None:
    headers = [
        "run",
        "step",
        "train_loss",
        "tps",
        "tflops",
        "mfu",
        "lpr_loss",
        "hit_rate",
        "load_cv",
        "active_frac",
        "entropy",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        widths["run"] = max(widths["run"], len(row["run"]))
        widths["step"] = max(widths["step"], len(str(row["global_step"])))
        widths["train_loss"] = max(widths["train_loss"], len(_format(row["train_loss"])))
        widths["tps"] = max(widths["tps"], len(_format(row["tps"])))
        widths["tflops"] = max(widths["tflops"], len(_format(row["tflops"])))
        widths["mfu"] = max(widths["mfu"], len(_format(row["mfu"])))
        widths["lpr_loss"] = max(widths["lpr_loss"], len(_format(row["lpr_loss"])))
        widths["hit_rate"] = max(widths["hit_rate"], len(_format(row["hit_rate"])))
        widths["load_cv"] = max(widths["load_cv"], len(_format(row["load_cv"])))
        widths["active_frac"] = max(widths["active_frac"], len(_format(row["active_frac"])))
        widths["entropy"] = max(widths["entropy"], len(_format(row["entropy"])))

    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(
            "  ".join(
                [
                    row["run"].ljust(widths["run"]),
                    str(row["global_step"]).ljust(widths["step"]),
                    _format(row["train_loss"]).ljust(widths["train_loss"]),
                    _format(row["tps"]).ljust(widths["tps"]),
                    _format(row["tflops"]).ljust(widths["tflops"]),
                    _format(row["mfu"]).ljust(widths["mfu"]),
                    _format(row["lpr_loss"]).ljust(widths["lpr_loss"]),
                    _format(row["hit_rate"]).ljust(widths["hit_rate"]),
                    _format(row["load_cv"]).ljust(widths["load_cv"]),
                    _format(row["active_frac"]).ljust(widths["active_frac"]),
                    _format(row["entropy"]).ljust(widths["entropy"]),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ACL smoke training outputs.")
    parser.add_argument("--root", default="outputs/acl_smoke", help="Root output directory.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Output root not found: {root}")

    runs = _collect_runs(root)
    if not runs:
        raise SystemExit(f"No completed runs found in {root}")

    _print_table(runs)


if __name__ == "__main__":
    main()
