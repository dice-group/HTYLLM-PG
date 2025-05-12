import glob
import wandb

from transformers import TrainerCallback
import subprocess, json, os
from pathlib import Path

class LMEvalCallback(TrainerCallback):
    def __init__(self, base_model_path ,tokenizer_name, eval_interval=500, eval_tasks=None, output_dir="./lm_eval_results", tb_logdir=None):
        self.eval_interval = eval_interval
        self.base_model_path = base_model_path
        self.eval_tasks = eval_tasks or ["hellaswag"]
        self.tokenizer_name = tokenizer_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_interval != 0 or state.global_step == 0:
            return control

        step_dir = self.output_dir / f"step_{state.global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # start eval subprocess based on latest checkpoint
        model_path = self.get_latest_checkpoint(args.output_dir)
        subprocess.run([
            "lm_eval",
            "--model", "hf",
            #"--model_args", f"pretrained={model_path},tokenizer={self.tokenizer_name}",
            "--model_args", f"pretrained={self.base_model_path},tokenizer={self.tokenizer_name},peft={model_path}",
            "--tasks", ",".join(self.eval_tasks),
            "--batch_size", "2",
            "--limit", "10",
            "--output_path", str(step_dir),
            "--wandb_args", f"project=your_project_name,group=eval,job_type=step_{state.global_step}",
            "--log_samples"
        ], check=True)

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
        wandb.log(log_data, step=state.global_step)

    def get_latest_checkpoint(self, output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return output_dir
        return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    def on_train_end(self, *args, **kwargs):
        self.tb_writer.close()
