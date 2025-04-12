import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


def data_collator(batch, tokenizer):
    input_ids = [torch.tensor(sample["tokens"], dtype=torch.long) for sample in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": (input_ids != tokenizer.pad_token_id).long()
    }


def main():
    parser = argparse.ArgumentParser(description="Train LLM on preprocessed chunks")
    parser.add_argument("--processed_dir", type=str, default=os.getenv("OUTPUT_DIR", "./output"),
                        help="Directory of preprocessed JSON chunks")
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_model_dir", type=str, default="./model_output")
    args = parser.parse_args()

    proc_rank = int(os.getenv("PROC_RANK", "0"))
    total_procs = int(os.getenv("TOTAL_PROCS", "1"))
    
    data_files = os.path.join(args.processed_dir, f"chunk_{proc_rank}_*.json")
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    def add_labels(example):
        example["labels"] = example["tokens"]
        return example
    dataset = dataset.map(add_labels, num_proc=16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    training_args = TrainingArguments(
        output_dir=args.output_model_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=25,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True if torch.cuda.is_available() else False,
        dataloader_num_workers=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: data_collator(batch, tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_model_dir)

if __name__ == "__main__":
    main()
