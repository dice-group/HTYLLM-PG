"""Pre‑train tiny Mixtral from JSONL(GZ) or TXT corpora using HF‑Trainer + DeepSpeed."""

from __future__ import annotations

import gzip, json
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    model_path: str = "checkpoints/init"
    tokenizer_path: str = "tokenizer"
    data_glob: str = "data/corpus/*"
    output_dir: str = "checkpoints/pretrain-run"

    seq_length: int = 1024
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 3e-4
    epochs: int = 3
    deepspeed_config: str = "configs/deepspeed/ds_zero3_moe.json"
    json_field: str = "text"  # field that holds the document text


# ---------------- tokenisation helper ----------------

def tokenize_and_chunk(text: str, tokenizer, seq_len: int):
    """Yield fixed‑length input‑id chunks from a raw text string."""
    ids = tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    for i in range(0, len(ids) - seq_len, seq_len):
        yield {"input_ids": ids[i : i + seq_len]}


# --------------- dataset loader ----------------------

def get_streaming_dataset(files: list[str], field: str):
    """Return a streaming HF dataset with a single `text` column."""
    # if extension is txt → use 'text' loader, else use 'json'
    sample_ext = Path(files[0]).suffix.lower()
    if sample_ext in {".txt", ".text"}:
        ds = load_dataset("text", data_files=files, split="train", streaming=True)
    else:
        ds = load_dataset("json", data_files=files, split="train", streaming=True)
        ds = ds.rename_column(field, "text")  # unify
    return ds


def main():
    parser = HfArgumentParser(ScriptArgs)
    (args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    files = [str(p) for p in Path().glob(args.data_glob)]
    if not files:
        raise SystemExit(f"No files match {args.data_glob}")

    raw = get_streaming_dataset(files, args.json_field)

    # map each document → sequence chunks lazily
    token_ds = raw.map(lambda ex: tokenize_and_chunk(ex["text"], tokenizer, args.seq_length), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model = MixtralForCausalLM.from_pretrained(args.model_path)

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

    trainer = Trainer(model=model, args=targs, train_dataset=token_ds, data_collator=data_collator)
    trainer.train()


if __name__ == "__main__":
    main()
