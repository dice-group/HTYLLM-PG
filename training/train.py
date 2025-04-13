import os
import argparse
import torch

from glob import glob
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, PhiForCausalLM, PhiConfig

def data_collator(batch, tokenizer):
    input_ids = [torch.tensor(sample["tokens"], dtype=torch.long) for sample in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default=os.getenv("INPUT_DIR", "./output"))
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_model_dir", type=str, default="./model_output")
    args = parser.parse_args()

    proc_rank = int(os.getenv("PROC_RANK", "0"))
    total_procs = int(os.getenv("TOTAL_PROCS", "1"))
    
    data_files = glob(os.path.join(args.processed_dir, "*", "shard_*.json"))
    print(f"Loading {len(data_files)} preprocessed files")
    #for file in data_files:
        #print(f" - {file}")
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    def add_labels(example):
        example["labels"] = example["tokens"]
        return example
    dataset = dataset.map(add_labels, num_proc=16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    config = PhiConfig(
        vocab_size=len(tokenizer),
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=8192,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        initializer_range=0.02,
    )
    model = PhiForCausalLM(config)

    training_args = TrainingArguments(
        output_dir=args.output_model_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: data_collator(batch, tokenizer)
    )

    trainer.train()
    trainer.save_model(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

if __name__ == "__main__":
    main()
