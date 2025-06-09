#!/usr/bin/env python
"""
Fine-tune / pre-train from an *already tokenised* dataset.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

# ── local HF cache so shared filesystems don't complain ────────────────────
cache_dir = Path("./cache/huggingface_datasets").absolute()
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)
os.environ["HF_HOME"] = str(cache_dir.parent)

# ── libraries ──────────────────────────────────────────────────────────────
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    MixtralForCausalLM,
)
from harness_callback import LMEvalCallback   # your eval harness

# ────────────────────────────────────────────────────────────────────────────
@dataclass
class ScriptArgs:
    dataset_dir: str
    model_path: str = "checkpoints/init"
    tokenizer_path: str = "tokenizer"
    output_dir: str = "checkpoints/pretrain-run"

    batch_size: int = 32
    grad_accum: int = 4
    lr: float = 3e-4
    epochs: int = 1
    max_steps: int = -1
    logging_steps: int = 50
    deepspeed_config: str | None = None      # ZeRO, profiler, …
    resume_from_checkpoint: str | None = None  # ← explicit CLI flag

# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args, = HfArgumentParser(ScriptArgs).parse_args_into_dataclasses()

    # tokenizer & dataset ----------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    ds  = load_from_disk(args.dataset_dir)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # model ------------------------------------------------------------------
    model = MixtralForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()

    # training args ----------------------------------------------------------
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

    # callbacks --------------------------------------------------------------
    callbacks = [LMEvalCallback(
        model=model,
        tokenizer=tok,
        task_list=("hellaswag", "mmlu", "belebele"),
        fewshot=0,
        limit=5,
        batch_size=16,
        prefix="harness",
    )]

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        callbacks=callbacks,
        tokenizer=tok,
    )

    # train ------------------------------------------------------------------
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
