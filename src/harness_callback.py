# harness_callback.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence


import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# ---- lm-harness imports -----------------------------------------------------
from lm_eval import evaluator, tasks          # v0.4 interface  :contentReference[oaicite:0]{index=0}
from lm_eval.models.huggingface import HFLM   # wrapper around any HF AutoModel

class LMEvalCallback(TrainerCallback):
    """
    Run lm-evaluation-harness at the end of every epoch and log the scores
    back into the Trainer logs (so they show up in TensorBoard/W&B/etc.).
    """

    def __init__(
        self,
        model,                      # the HF model instance
        tokenizer,                  # the HF tokenizer instance
        task_list: Sequence[str] = ("hellaswag", "mmlu", "belebele"), 
        fewshot: int = 0,
        limit: int | None = None,
        batch_size: int = 16,
        prefix: str = "harness",
    ) -> None:
        # Store model and tokenizer for use in on_epoch_end
        self.model = model
        self.tokenizer = tokenizer
        self.task_list = list(task_list)
        self.fewshot = fewshot
        self.limit = limit
        self.batch_size = batch_size
        self.prefix = prefix

    # We hook into on_epoch_end but you could also use on_save or on_step_end
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,  # Accept any additional kwargs that Trainer might pass
    ):
        # Only run evaluation on main process in distributed training
        if args.local_rank not in [-1, 0]:
            return control
            
        # Wrap current checkpoint for lm-eval using our stored model/tokenizer
        lm = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=self.batch_size,
        )

        # Run the harness
        results = evaluator.evaluate(
            lm,
            task_dict=tasks.get_task_dict(self.task_list),
            num_fewshot=self.fewshot,
            limit=self.limit,
        )

        # Flatten and push into Trainer's log stream
        flat = {
            f"{self.prefix}/{k}": v
            for k, v in results["results"].items()
        }
        control.metrics.update(flat)      # ensures logs & callbacks see the numbers
        control.should_log = True
        return control
