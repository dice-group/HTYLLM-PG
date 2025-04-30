"""Lightweight Trainer script for tiny Mixtral using HuggingFace + DeepSpeed."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from datasets import load_dataset
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
    # paths / names
    model_path: str = "checkpoints/init"
    tokenizer_path: str = "tokenizer"
    data_glob: str = "data/processed/*.txt"
    output_dir: str = "checkpoints/pretrain-run"

    # training
    seq_length: int = 1024
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 3e-4
    epochs: int = 3
    deepspeed_config: str = "configs/deepspeed/ds_zero3_moe.json"


# --------------------------------------------------------------------------

def tokenize_and_chunk(stream, tokenizer, seq_len):
    """Concatenate text lines then chunk to fixed length token IDs."""
    buffer: List[int] = []
    for item in stream:
        ids = tokenizer(item["text"], add_special_tokens=False).input_ids
        buffer.extend(ids + [tokenizer.eos_token_id])
        while len(buffer) >= seq_len:
            chunk, buffer = buffer[:seq_len], buffer[seq_len:]
            yield {"input_ids": chunk}


def main():
    parser = HfArgumentParser(ScriptArgs)
    (args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # ------------------------------------------------------------------ DATA
    raw = load_dataset("text", data_files=list(Path().glob(args.data_glob)), split="train", streaming=True)
    token_ds = raw.map(lambda x: tokenize_and_chunk([x], tokenizer, args.seq_length))

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # --------------------------------------------------------------- MODEL
    model = MixtralForCausalLM.from_pretrained(args.model_path)

    # ----------------------------------------------------------- TRAINER
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=1000,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=token_ds,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()