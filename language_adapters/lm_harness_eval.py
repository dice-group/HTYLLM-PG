import glob
import wandb

from transformers import TrainerCallback
import subprocess, json, os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class LMEvalCallback(TrainerCallback):
    def __init__(self, tokenizer_name, eval_interval=500, eval_tasks=None, output_dir="./lm_eval_results", tb_logdir=None, batch_size=2, limit=500, cuda_devices="0", wandb_project="lm_eval_project"):
        
        self.eval_interval = eval_interval
        self.eval_tasks = eval_tasks or ["belebele"]
        self.tokenizer_name = tokenizer_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_logdir) if tb_logdir else None
        
        self.batch_size = batch_size
        self.limit = limit
        self.cuda_devices = cuda_devices
        self.wandb_project = wandb_project

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_interval != 0 or state.global_step == 0:
            return control

        try:
            step_dir = self.output_dir / f"step_{state.global_step}"
            step_dir.mkdir(parents=True, exist_ok=True)

            # start eval subprocess based on latest checkpoint
            model_path = self.get_latest_checkpoint(args.output_dir)
            subprocess.run([
                "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={self.tokenizer_name},peft={model_path},tokenizer={self.tokenizer_name}",
                "--tasks", ",".join(self.eval_tasks),
                "--batch_size", str(self.batch_size),
                "--limit", str(self.limit),
                "--output_path", str(step_dir),
                "--wandb_args", f"project={self.wandb_project},group=eval,job_type=step_{state.global_step}",
                "--log_samples"
            ], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": self.cuda_devices})

            result_files = list(step_dir.glob("*/results_*.json"))
            if not result_files:
                raise FileNotFoundError(f"No results found in {step_dir}")

            with open(result_files[0]) as f:
                results = json.load(f)["results"]

            log_data = {}
            for task, metrics in results.items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        tag = f"{task}/{k.replace(',', '_')}"
                        log_data[tag] = v
                        if self.tb_writer:
                            self.tb_writer.add_scalar(tag, v, global_step=state.global_step)
            wandb.log(log_data, step=state.global_step)
            
        except Exception as e:
            print(f"[LMEvalCallback] Evaluation step failed at global_step {state.global_step}: {e}")

        return control

    def get_latest_checkpoint(self, output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return output_dir
        return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    def on_train_end(self, *args, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()


class LMHarnessEarlyStoppingCallback(TrainerCallback):
    def __init__(self, eval_dir, metric_names, patience=3):
        self.eval_dir = eval_dir
        self.metric_names = metric_names  # e.g., ["arc_zh", "belebele", "belebele_acm_Arab"] here select only those on which we want to improve, not all e.g only specific languages
        self.patience = patience
        self.best_score = None
        self.bad_evals = 0

    def on_evaluate(self, args, state, control, **kwargs):
        try:
            combined_score = self._read_and_sum_metrics()
            print(f"[LMHarnessEarlyStopping] Combined eval score: {combined_score:.4f}")

            if self.best_score is None or combined_score > self.best_score:
                print("[LMHarnessEarlyStopping] Improvement detected.")
                self.best_score = combined_score
                self.bad_evals = 0
            else:
                self.bad_evals += 1
                print(f"[LMHarnessEarlyStopping] No improvement. Patience: {self.bad_evals}/{self.patience}")

            if self.bad_evals >= self.patience:
                print("[LMHarnessEarlyStopping] Early stopping triggered.")
                control.should_training_stop = True

        except Exception as e:
            print(f"[LMHarnessEarlyStopping] Failed during evaluation check: {e}")

        return control

    def _read_and_sum_metrics(self):
        try:
            ## get latest step dir
            step_dirs = sorted(glob.glob(os.path.join(self.eval_dir, "step_*")), key=os.path.getmtime)
            latest_step_dir = step_dirs[-1]
            subdirs = [d for d in glob.glob(os.path.join(latest_step_dir, "*")) if os.path.isdir(d)]

            eval_dir = subdirs[0]  # assume one eval per step TODO: maybe make this more reliable
            result_files = glob.glob(os.path.join(eval_dir, "results_*.json"))
            if not result_files:
                print(f"[LMHarnessEarlyStopping] No results found in {eval_dir}")
                return 0.0

            latest_result_file = sorted(result_files)[-1]   #only one should exist per step file
            with open(latest_result_file, "r") as f:
                results = json.load(f)

            result_data = results.get("results", {})
            total_score = 0.0
            for task in self.metric_names:
                task_data = result_data.get(task, {})
                acc = task_data.get("acc,none", 0.0)
                total_score += acc
                
            ## write combined scores and individual scores in log file
            with open(os.path.join(self.eval_dir, "combined_scores.txt"), "a") as f:
                f.write(f"{latest_result_file}\n")
                for task in self.metric_names:
                    acc = result_data.get(task, {}).get("acc,none", 0.0)
                    f.write(f"  {task}: {acc:.4f}\n")
                f.write(f"  Total Combined Score: {total_score:.4f}\n\n")

            return total_score
        except Exception as e:
            print(f"[LMHarnessEarlyStopping] Error reading metrics: {e}")
            return 0.0