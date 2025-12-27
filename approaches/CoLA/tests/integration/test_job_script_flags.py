from __future__ import annotations

from pathlib import Path


def _assert_flag_present(path: Path, flag: str) -> None:
    content = path.read_text(encoding="utf-8")
    assert flag in content, f"Expected flag '{flag}' in {path}"


def test_job_scripts_include_throughput_flags() -> None:
    base = Path("scripts/comparison")
    scripts = [
        base / "cola_lpr_job.sh",
        base / "hydralora_lpr_job.sh",
        base / "lora_job.sh",
    ]
    for script_path in scripts:
        _assert_flag_present(script_path, "--include_effective_tokens_per_second true")
        _assert_flag_present(script_path, "--include_num_input_tokens_seen true")
