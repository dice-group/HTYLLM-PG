# harness_callback.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import os

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
        # Set datasets cache to local directory to avoid permission issues (only on rank 0)
        if args.local_rank in [-1, 0]:
            cache_dir = Path("./cache/huggingface_datasets").absolute()
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Set multiple environment variables to ensure cache directory is used
            os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
            os.environ["DATASETS_CACHE"] = str(cache_dir)  
            os.environ["HF_HOME"] = str(cache_dir.parent)
            
            # Also try to set via datasets library if available
            try:
                import datasets
                datasets.config.CACHE_DIR = str(cache_dir)
            except:
                pass
        
        # Increase NCCL timeout to prevent timeouts during evaluation
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" 
        # Set timeout to 30 minutes (1800 seconds)
        os.environ["NCCL_TIMEOUT"] = "1800"
        
        # Synchronize all processes before starting evaluation
        if args.local_rank != -1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        
        try:
            # Wrap current checkpoint for lm-eval using our stored model/tokenizer
            # This will automatically use distributed evaluation if multiple GPUs are available
            lm = HFLM(
                pretrained=self.model,
                tokenizer=self.tokenizer,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=self.batch_size,
            )

            # Run the harness (will distribute across available GPUs)
            results = evaluator.evaluate(
                lm,
                task_dict=tasks.get_task_dict(self.task_list),
                limit=self.limit,
            )

            # Only log results on rank 0 to avoid duplicate logging
            if args.local_rank in [-1, 0]:
                # Flatten and push into Trainer's log stream
                flat = {
                    f"{self.prefix}/{k}": v
                    for k, v in results["results"].items()
                }
                control.metrics.update(flat)      # ensures logs & callbacks see the numbers
                control.should_log = True
            
        except Exception as e:
            if args.local_rank in [-1, 0]:
                print(f"Evaluation failed: {e}")
            # Continue training even if evaluation fails
            
        # Synchronize all processes after evaluation
        if args.local_rank != -1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
            
        return control
