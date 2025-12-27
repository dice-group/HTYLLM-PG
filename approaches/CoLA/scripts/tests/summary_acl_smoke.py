from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        run = {
            "run": output_dir.name,
            "global_step": state.get("global_step", 0),
            "train_loss": results.get("train_loss"),
            "train_runtime": results.get("train_runtime"),
            "tps": results.get("effective_tokens_per_sec"),
            "lpr_loss": _find_last_metric(history, "language_prior_loss"),
            "router_metric": _find_last_prefix(history, "cola/")
            or _find_last_prefix(history, "hydralora/"),
        }
        runs.append(run)
    return runs


def _print_table(rows: list[dict]) -> None:
    headers = ["run", "step", "train_loss", "tps", "lpr_loss", "router_metric"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        widths["run"] = max(widths["run"], len(row["run"]))
        widths["step"] = max(widths["step"], len(str(row["global_step"])))
        widths["train_loss"] = max(widths["train_loss"], len(_format(row["train_loss"])))
        widths["tps"] = max(widths["tps"], len(_format(row["tps"])))
        widths["lpr_loss"] = max(widths["lpr_loss"], len(_format(row["lpr_loss"])))
        metric = row["router_metric"]
        metric_str = "-" if metric is None else f"{metric[0]}={_format(metric[1])}"
        widths["router_metric"] = max(widths["router_metric"], len(metric_str))

    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        metric = row["router_metric"]
        metric_str = "-" if metric is None else f"{metric[0]}={_format(metric[1])}"
        print(
            "  ".join(
                [
                    row["run"].ljust(widths["run"]),
                    str(row["global_step"]).ljust(widths["step"]),
                    _format(row["train_loss"]).ljust(widths["train_loss"]),
                    _format(row["tps"]).ljust(widths["tps"]),
                    _format(row["lpr_loss"]).ljust(widths["lpr_loss"]),
                    metric_str.ljust(widths["router_metric"]),
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
