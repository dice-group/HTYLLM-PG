import os
import glob
import subprocess
import json
import argparse
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
import torch

# ======= seed for reproducibility =======
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

for ckpt_path in checkpoints:
    step = int(ckpt_path.split("-")[-1])
    step_dir = os.path.join(eval_output_base, f"step_{step}")
    Path(step_dir).mkdir(parents=True, exist_ok=True)

    print(f"Evaluating checkpoint at step {step}...")

    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    run_env["WANDB_PROJECT"] = args.wandb_project
    run_env["WANDB_RUN_ID"] = run_id
    run_env["WANDB_RESUME"] = "allow"

    subprocess.run([
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model_name},peft={ckpt_path},tokenizer={args.tokenizer_name}",
        "--tasks", args.eval_tasks,
        "--batch_size", args.batch_size,
        # "--limit", args.limit,  # Uncomment if needed
        "--output_path", step_dir,
        #"--wandb_args", f"project={args.wandb_project},group={args.wandb_group},job_type=step_{step}",
        "--log_samples"
    ], env=run_env, check=True)

    result_files = list(Path(step_dir).glob("*/results_*.json"))
    if not result_files:
        print(f"No results found in {step_dir}")
        continue

    with open(result_files[0]) as f:
        results = json.load(f)["results"]

    log_data = {}
    for task, metrics in results.items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                tag = f"{task}/{k.replace(',', '_')}"
                log_data[tag] = v
                tb_writer.add_scalar(tag, v, global_step=step)

    wandb.log(log_data, step=step)

# ======= finish =======
tb_writer.close()
wandb.finish()
print("Eval completed for all checkpoints.")
