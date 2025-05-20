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
from harness_callback import LMEvalCallback

@dataclass
class ScriptArgs:
    dataset_dir: str            # <-- NEW: path produced by preprocess.py
    model_path: str = "checkpoints/init"
    tokenizer_path: str = "tokenizer"  # <-- Added tokenizer path parameter
    output_dir: str = "checkpoints/pretrain-run"

    batch_size: int = 24
    grad_accum: int = 4
    lr: float = 3e-4
    epochs: int = 3
    max_steps: int = -1
    logging_steps: int = 50
    deepspeed_config: str | None = None


def main():
    args, = HfArgumentParser(ScriptArgs).parse_args_into_dataclasses()
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)  # <-- Use the tokenizer path directly

    # ── Load the arrow dataset (zero‑copy memory‑mapped) ────────────────────
    ds = load_from_disk(args.dataset_dir)
    # (optional) shuffle each epoch via Trainer's dataloader, or:
    # ds = ds.shuffle(seed=42)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    model = MixtralForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()
    #model.enable_flash_attention_2d()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_steps=1_000,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
    )

    callbacks = [LMEvalCallback(
        task_list=("hellaswag", "mmlu", "belebele"),
        fewshot=0,
        batch_size=16,
        prefix="harness",
    )]

    Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        callbacks=callbacks,
    ).train()
    


if __name__ == "__main__":
    main()