from pathlib import Path
import json
import os
import sys

os.environ["WANDB_MODE"] = "online"

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

from scripts import lm_eval_language_ids as lm  # noqa: E402

out = Path("/tmp/lm_eval_fake")
out.mkdir(parents=True, exist_ok=True)


def write_fake(step: int, acc_no: float, acc_with: float) -> None:
    no_ids = {
        "results": {
            "belebele_eng_Latn": {"acc": acc_no},
            "belebele_deu_Latn": {"acc": acc_no - 0.02},
        }
    }
    (out / "no_language_ids.json").write_text(json.dumps(no_ids, indent=2))

    with_eng = {"results": {"belebele_eng_Latn": {"acc": acc_with}}}
    with_deu = {"results": {"belebele_deu_Latn": {"acc": acc_with - 0.01}}}
    (out / "with_language_ids_belebele_eng_Latn.json").write_text(json.dumps(with_eng, indent=2))
    (out / "with_language_ids_belebele_deu_Latn.json").write_text(json.dumps(with_deu, indent=2))

    # Logging handled by the caller to keep a single run per series.


import wandb

tiers = {
    "cola_tier_xy": (
        (100, 0.30, 0.35),
        (200, 0.34, 0.40),
        (300, 0.31, 0.36),
        (400, 0.33, 0.41),
    ),
    "cola_tier_xy2": (
        (100, 0.28, 0.33),
        (200, 0.31, 0.37),
        (300, 0.29, 0.34),
        (400, 0.32, 0.38),
    ),
}

for base_name, series in tiers.items():
    run_no = wandb.init(
        project="htyllm-lm-eval_summary",
        name=f"{base_name}_no_ids",
        id=wandb.util.generate_id(),
        resume="allow",
        mode="online",
    )
    for step, acc_no, acc_with in series:
        write_fake(step, acc_no, acc_with)
        run_no.log(
            {
                "belebele_eng_Latn/acc": acc_no,
                "belebele_deu_Latn/acc": acc_no - 0.02,
            },
            step=step,
        )
    run_no.finish()

    run_with = wandb.init(
        project="htyllm-lm-eval_summary",
        name=f"{base_name}_with_ids",
        id=wandb.util.generate_id(),
        resume="allow",
        mode="online",
    )
    for step, acc_no, acc_with in series:
        run_with.log(
            {
                "belebele_eng_Latn/acc": acc_with,
                "belebele_deu_Latn/acc": acc_with - 0.01,
            },
            step=step,
        )
    run_with.finish()
