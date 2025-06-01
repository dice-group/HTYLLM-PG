import os
import sys
import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from lm_eval import evaluator
import torch._dynamo


def get_checkpoints(checkpoint_dir):
    return sorted([
        str(p) for p in Path(checkpoint_dir).iterdir()
        if p.is_dir() and "checkpoint" in p.name
    ])


def evaluate_checkpoint(checkpoint_path, output_dir):
    print(f"Evaluating checkpoint: {checkpoint_path}")
    torch._dynamo.disable()
    checkpoint_name = Path(checkpoint_path).name
    results_dir = os.path.join(output_dir, "lm_eval", checkpoint_name)
    tb_logdir = os.path.join(output_dir, "runs", "lm_eval", checkpoint_name)
    os.makedirs(results_dir, exist_ok=True)

    # Run evaluation using lm-eval-harness
    results = evaluator.simple_evaluate(
        model="hf",
        model_args={
            "pretrained": checkpoint_path,
            "tokenizer": "google/gemma-3-4b-pt",
            "revision": "main"
        },
        tasks=[
            "hellaswag", "xnli", "belebele", "arc_multilingual", "truthfulqa", "mgsm_direct", "mgsm_cot_native", "xcopa", "xwinograd", "xstorycloze", "xnli", "pawsx", "flores", "wmt16", "lambada_multilingual", "xquad"  # add any further eval_tasks here
        ],
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
        limit=100
    )

    # Manually save results to JSON
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Finished evaluating {checkpoint_path}. Logging to TensorBoard...")

    # --- TensorBoard logging ---
    writer = SummaryWriter(log_dir=tb_logdir)
    for task_name, task_result in results["results"].items():
        for metric, value in task_result.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(f"{task_name}/{metric}", value, 0)
    writer.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python lm_eval_runner.py <path_to_checkpoints_dir>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]

    if not Path(checkpoint_dir).is_dir():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    checkpoints = get_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints. Starting evaluation...")

    for ckpt in checkpoints:
        evaluate_checkpoint(ckpt, checkpoint_dir + "/..")

    print("All evaluations completed.")


if __name__ == "__main__":
    main()

