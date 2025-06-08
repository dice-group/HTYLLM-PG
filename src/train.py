#!/usr/bin/env python
"""
Fine‑tune / pre‑train from an *already tokenised* dataset.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

# Set datasets cache to local directory to avoid permission issues
cache_dir = Path("./cache/huggingface_datasets").absolute()
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)
os.environ["HF_HOME"] = str(cache_dir.parent)

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
    TrainerCallback,
)
from flops_profiler import get_model_profile
from harness_callback import LMEvalCallback


@dataclass
class ScriptArgs:
    dataset_dir: str            # <-- NEW: path produced by preprocess.py
    model_path: str = "checkpoints/init"
    tokenizer_path: str = "tokenizer"  # <-- Added tokenizer path parameter
    output_dir: str = "checkpoints/pretrain-run"

    batch_size: int = 32
    grad_accum: int = 4
    lr: float = 3e-4
    epochs: int = 1
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
        model=model,
        tokenizer=tok,
        task_list=("hellaswag", "mmlu", "belebele"),
        fewshot=0,
        limit=5,
        batch_size=16,
        prefix="harness",
    )]

    # ────────────────────── FLOPs odometer callback ───────────────────────
    class FlopsOdometer(TrainerCallback):
        """
        • Measures FLOPs for *one* real training batch (fwd+back+opt).
        • Displays cumulative FLOPs & token count at every log step.
        """

        def __init__(self, tokenizer, seq_len: int = 2048):
            self.tok = tokenizer
            self.seq_len = seq_len
            self.per_step_flops = None          # lazily filled
            self.per_step_tokens = None

        # ── run once, on first forward/backward step ───────────────────────
        def on_step_end(self, args, state, control, **kwargs):
            if self.per_step_flops is not None:
                return

            model = kwargs["model"].module if hasattr(kwargs["model"], "module") else kwargs["model"]
            batch = kwargs["inputs"]["input_ids"]            # real batch, already on device

            # Run a *single* profiling pass (no grads so we capture only fwd)
            flops_fwd, macs, _ = get_model_profile(
                model=model,
                input_shape=batch.shape,
                print_profile=False,
                detailed=False,
            )
            # crude but typical multiplier: bwd ≈ 2× fwd, opt ≈ 1× fwd
            self.per_step_flops = flops_fwd * 3
            self.per_step_tokens = batch.numel()

            if state.is_local_process_zero:
                gf = self.per_step_flops / 1e9
                print(f"[FLOPs-Profiler] 1 training step ≈ {gf:.2f} GFLOPs "
                      f"for {batch.size(0)}×{batch.size(1)} tokens")

        # ── emit rolling totals every logging interval ─────────────────────
        def on_log(self, args, state, control, **kwargs):
            if self.per_step_flops is None:
                return
            steps = state.global_step
            total_flops = self.per_step_flops * steps
            total_tokens = self.per_step_tokens * steps
            if state.is_local_process_zero:
                print(f"[FLOPs-Profiler] so far: "
                      f"{total_flops/1e15:.3f} PFLOPs over {total_tokens/1e9:.2f} B tokens")

        # ── final summary ──────────────────────────────────────────────────
        def on_train_end(self, args, state, control, **kwargs):
            if self.per_step_flops is None:
                return
            steps = state.global_step
            total_flops = self.per_step_flops * steps
            if state.is_local_process_zero:
                print(f"\n=== Training complete ===")
                print(f"Total training steps : {steps}")
                print(f"Total tokens seen    : {self.per_step_tokens*steps:,}")
                print(f"Total compute used   : {total_flops/1e15:.3f} PFLOPs\n")

    callbacks.append(FlopsOdometer(tok))

    Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        callbacks=callbacks,
        tokenizer=tok,
    ).train(resume_from_checkpoint=args.model_path)


if __name__ == "__main__":
    main()