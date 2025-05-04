"""Pre‑train tiny Mixtral from JSONL(GZ) or TXT corpora using HF‑Trainer + DeepSpeed."""

from __future__ import annotations

import gzip, json, logging
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
    deepspeed_config: str | None = None
    json_field: str = "text"  # field that holds the document text
    max_steps: int = -1  # Add max_steps
    logging_steps: int = 50  # Add logging_steps


# ---------------- tokenisation helper ----------------

def tokenize_and_chunk(text: str, tokenizer, seq_len: int):
    """Tokenizes text and returns a dictionary {'input_ids': [chunk1, chunk2, ...]} of full-length chunks."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return {"input_ids": []}  # empty text → return empty list
    ids.append(tokenizer.eos_token_id)

    chunks = []
    for i in range(0, len(ids), seq_len):  # include final chunk for potential slicing
        chunk = ids[i : i + seq_len]
        if len(chunk) == seq_len:  # keep only full-length chunks
            chunks.append(chunk)
    return {"input_ids": chunks}


# --------------- dataset loader ----------------------

def get_streaming_dataset(files: list[str], field: str):
    """Return a streaming HF dataset with a single `text` column."""
    # if extension is txt → use 'text' loader, else use 'json'
    sample_ext = Path(files[0]).suffix.lower()
    if sample_ext in {'.txt', '.text'}:
        ds = load_dataset("text", data_files=files, split="train", streaming=True)
    else:
        ds = load_dataset("json", data_files=files, split="train", streaming=True)
        # Only rename if the field is not already 'text'
        if field != 'text':
            ds = ds.rename_column(field, "text")  # unify
    return ds


def main():
    parser = HfArgumentParser(ScriptArgs)
    (args,) = parser.parse_args_into_dataclasses()

    # --- Basic logging setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # --- End logging setup ---

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    files = [str(p) for p in Path().glob(args.data_glob)]
    if not files:
        raise SystemExit(f"No files match {args.data_glob}")

    raw = get_streaming_dataset(files, args.json_field)
    logging.info(f"Initial raw dataset type: {type(raw)}")
    raw = raw.filter(
    lambda ex: len(tokenizer(ex["text"], add_special_tokens=False).input_ids) + 1 
               >= args.seq_length
    )
    # 2) batched chunker (always returns ≥1 chunk)
    def chunk_batch(batch):
        all_ids = []
        for text in batch["text"]:
            ids = tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            total_len = (len(ids) // args.seq_length) * args.seq_length
            for i in range(0, total_len, args.seq_length):
                all_ids.append(ids[i : i + args.seq_length])
        return {"input_ids": all_ids}   

    token_ds = raw.map(
        chunk_batch,
        batched=True,
        batch_size=1,            # one raw example at a time
        remove_columns=["text"],
    )



    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model = MixtralForCausalLM.from_pretrained(args.model_path)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,  # Use max_steps from ScriptArgs
        bf16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,  # Use logging_steps from ScriptArgs
        save_steps=1000,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed_config if args.deepspeed_config else None,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=token_ds, data_collator=data_collator)
    trainer.train()


if __name__ == "__main__":
    main()
