import os
import json
import argparse
import subprocess
import random
import numpy as np
import torch
import wandb

from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# --- Set deterministic seed ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--tokenizer_name", required=True)
parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--tasks", required=True)
parser.add_argument("--batch_size", default="32")
parser.add_argument("--limit", default="50")
parser.add_argument("--wandb_project", default="eval_project")
parser.add_argument("--wandb_group", default="eval_group")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# --- Setup TensorBoard and WandB ---
tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
wandb.init(project=args.wandb_project, group=args.wandb_group, job_type="evaluation")

# --- Evaluate each task separately ---
summary_results = {}
tasks = args.tasks.split(",")

for task in tasks:
    print(f"\n=== Evaluating task: {task} ===")

    try:
        task_output_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)

        subprocess.run([
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={args.model_name},tokenizer={args.tokenizer_name}",
            "--tasks", task,
            "--batch_size", args.batch_size,
            "--limit", args.limit,
            "--output_path", task_output_dir,
            "--log_samples",
            "--seed", str(SEED)
        ], check=True)

        # --- Log results for this task ---
        result_files = list(Path(task_output_dir).glob("*/results_*.json"))
        if result_files:
            with open(result_files[0]) as f:
                results = json.load(f)["results"]
            for task_name, metrics in results.items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        tag = f"{task_name}/{k}"
                        tb_writer.add_scalar(tag, v, 0)
                        wandb.log({tag: v}, step=0)
                        if k == "acc,none":
                            summary_results[task_name] = v

    except subprocess.CalledProcessError as e:
        error_msg = f"[ERROR] Failed to evaluate task: {task} — Error: {e}"
        print(error_msg)
        with open(os.path.join(args.output_dir, "error_log.txt"), "a") as err_log:
            err_log.write(f"{datetime.now()} - {error_msg}\n")
        continue  # move to next task

# --- Finish logging ---
tb_writer.close()
wandb.finish()

# --- Print summary table ---
print("\n=== Summary of Accuracies ===")
df = pd.DataFrame(list(summary_results.items()), columns=["Task", "Accuracy"])
df = df.sort_values(by="Task")
print(df.to_markdown(index=False))

# --- Save summary to file ---
with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
    json.dump(summary_results, f, indent=2)

print("Evaluation complete.")
