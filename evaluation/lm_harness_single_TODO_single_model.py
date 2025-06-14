import os
import glob
import traceback
import subprocess
import json
import argparse
import wandb
import random
import numpy as np
import torch
import pandas as pd

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# ======= seed for reproducibility =======
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======= restrict cores =======
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ======= parser =======
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", required=True)
parser.add_argument("--tokenizer_name", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--eval_tasks", required=True, help="Comma-separated list")
parser.add_argument("--batch_size", default="32")
parser.add_argument("--limit", default="50")
parser.add_argument("--wandb_project", default="eval_project")
parser.add_argument("--wandb_group", default="eval_group")
parser.add_argument("--run_name", default="eval_run")
parser.add_argument("--cuda_devices", default="0")
parser.add_argument("--wandb_run_id", default="")
args = parser.parse_args()

# ======= setup =======
logdir = os.path.join(args.base_dir, "logs")
eval_output_base = Path(os.path.join(args.base_dir, "lm_eval"))
Path(logdir).mkdir(parents=True, exist_ok=True)
Path(eval_output_base).mkdir(parents=True, exist_ok=True)

tb_writer = SummaryWriter(log_dir=logdir)

# ======= wandb init (single run) =======
run = wandb.init(
    project=args.wandb_project,
    name=args.run_name,
    resume="allow"
)
run_id = wandb.run.id

# ======= tasks =======
summary_results = {}
eval_tasks = [task.strip() for task in args.eval_tasks.split(",")]
step            = 0  # constant so downstream logging code still works
step_dir = eval_output_base / f"step_{step}"
step_dir.mkdir(parents=True, exist_ok=True)
print(f"Evaluating base model '{args.model_name}' …")    
    
for task in eval_tasks:
    task_output_dir = os.path.join(step_dir, task)
    Path(task_output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n-- Task: {task} --")
    run_env = os.environ.copy()
    run_env["WANDB_PROJECT"] = args.wandb_project
    run_env["WANDB_RUN_ID"] = run_id
    run_env["WANDB_RESUME"] = "allow"

    try:
        subprocess.run([
            "lm_eval",
            "--model", "hf",
            "--model_args",
            f"pretrained={args.model_name},tokenizer={args.tokenizer_name}",
            "--tasks", task,
            "--batch_size", args.batch_size,
            "--limit", args.limit,
            "--output_path", str(task_output_dir),
            "--log_samples",
        ], env=run_env, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Evaluation failed for task '{task}'. Skipping …")
        traceback.print_exc()
        continue

    result_files = list(Path(task_output_dir).glob("results_*.json"))
    if not result_files:
        print(f"[WARNING] No result file found for task '{task}' at step {step}.")
        continue

    try:
        with open(result_files[0]) as f:
            results = json.load(f)["results"]

        log_data = {}
        for tsk, metrics in results.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    tag = f"{tsk}/{k.replace(',', '_')}"
                    log_data[tag] = v
                    tb_writer.add_scalar(tag, v, global_step=step)
                    if k == "acc,none":
                        summary_results[tsk] = v       # only one step
        wandb.log(log_data, step=step)
    except Exception as e:
        print(f"[ERROR] Failed to process results for task '{task}': {e}")
        traceback.print_exc()
        continue

# ======= finish =======
tb_writer.close()
wandb.finish()

# --- Print summary table ---
print("\n=== Summary of Accuracies ===")
df = pd.DataFrame(list(summary_results.items()), columns=["Task", "Accuracy"])
df = df.sort_values(by="Task")
print(df.to_markdown(index=False))

# --- Save summary to file ---
with open(os.path.join(eval_output_base, "summary.json"), "w") as f:
    json.dump(summary_results, f, indent=2)

print("Evaluation complete.")

#TODO: this script works nice in sense of sequentially evlauationg task after task, but the loggin to wandb should be improved
# TODO: automatically extraction of the results should be added in the end