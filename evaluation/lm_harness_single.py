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
eval_output_base = os.path.join(args.base_dir, "lm_eval")
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

# ======= checkpoint loop =======
checkpoints = sorted(
    glob.glob(os.path.join(args.base_dir, "checkpoint-*")),
    key=lambda x: int(x.split("-")[-1])
)

# ======= eval once for all tasks =======
summary_results = {}

for ckpt_path in checkpoints:
    step = int(ckpt_path.split("-")[-1])
    step_dir = os.path.join(eval_output_base, f"step_{step}")
    Path(step_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== Evaluating checkpoint at step {step} on all tasks ===")

    run_env = os.environ.copy()
    run_env["WANDB_PROJECT"] = args.wandb_project
    run_env["WANDB_RUN_ID"] = run_id
    run_env["WANDB_RESUME"] = "allow"

    try: 
        subprocess.run([
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={args.model_name},peft={ckpt_path},tokenizer={args.tokenizer_name}",
            "--tasks", args.eval_tasks,
            "--batch_size", args.batch_size,
            #"--limit", args.limit,
            "--output_path", step_dir,
            #"--wandb_args", f"project={args.wandb_project},group={args.wandb_group},job_type=step_{step}",
            "--log_samples"
        ], env=run_env, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Evaluation failed at step {step}. Skipping...")
        traceback.print_exc()
        continue
    

    result_files = list(Path(step_dir).glob("results_*.json"))
    if not result_files:
        print(f"[WARNING] No result file found at step {step}.")
        continue

    try:
        with open(result_files[0]) as f:
            results = json.load(f)["results"]

        log_data = {}
        for task, metrics in results.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    tag = f"{task}/{k.replace(',', '_')}"
                    log_data[tag] = v
                    tb_writer.add_scalar(tag, v, global_step=step)
                    if k == "acc,none":
                        summary_results[f"{task}@step{step}"] = v
        wandb.log(log_data, step=step)
    except Exception as e:
        print(f"[ERROR] Failed to process results at step {step}: {e}")
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