from transformers import TrainerCallback
import subprocess
import os
import glob

def get_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return output_dir
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest

class LMEvalCallback(TrainerCallback):
    def __init__(self, tokenizer_name, eval_interval=500, eval_tasks=None, output_dir="./lm_eval_results"):
        self.eval_interval = eval_interval
        self.eval_tasks = eval_tasks or ["hellaswag"]
        self.tokenizer_name = tokenizer_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_interval == 0 and state.global_step != 0:
            model_path = get_latest_checkpoint(args.output_dir)
            print(f"\n[LM Eval] Running LM Evaluation at step {state.global_step}...\n")

            command = [
                "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_path},tokenizer={self.tokenizer_name}",
                "--tasks", ",".join(self.eval_tasks),
                "--batch_size", "4",
                "--limit", "1000",
                "--output_path", f"{self.output_dir}/step_{state.global_step}.json",
            ]
            subprocess.run(command)
        return control
