#!/usr/bin/env python
"""
Fine‑tune / pre‑train from an *already tokenised* dataset.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    MixtralForCausalLM,
)


@dataclass
class ScriptArgs:
    dataset_dir: str            # <-- NEW: path produced by preprocess.py
    model_path: str = "checkpoints/init"
    tokenizer_path: str = "tokenizer"  # <-- Added tokenizer path parameter
    output_dir: str = "checkpoints/pretrain-run"

    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 3e-4
    epochs: int = 3
    max_steps: int = -1
    logging_steps: int = 50
    deepspeed_config: str | None = None


def main():
    args, = HfArgumentParser(ScriptArgs).parse_args_into_dataclasses()
    
    # Initialize distributed environment if run with torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    is_distributed = local_rank != -1
    
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        print(f"Initialized process {local_rank} / {world_size}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)  # <-- Use the tokenizer path directly

    # ── Load the arrow dataset (zero‑copy memory‑mapped) ────────────────────
    ds = load_from_disk(args.dataset_dir)
    # (optional) shuffle each epoch via Trainer's dataloader, or:
    # ds = ds.shuffle(seed=42)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    model = MixtralForCausalLM.from_pretrained(args.model_path)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=1_000,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed_config,
        # Distributed training parameters
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
    )

    Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    ).train()
    
    # Clean up process group for distributed training
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()