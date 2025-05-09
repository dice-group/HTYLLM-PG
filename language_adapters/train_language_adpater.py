import os
import gzip
import argparse
import torch

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from lm_harness_eval import LMEvalCallback

def read_gz_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                path = os.path.join(root, file)
                with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and len(line) > 10:
                            yield {"text": line}


def tokenize_data(data_dir, model_name, tokenized_output_dir):
    print(f"Tokenizing data from {data_dir} using tokenizer {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = Dataset.from_generator(lambda: read_gz_folder(data_dir))
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=512),
        batched=True, num_proc=24
    )
    tokenized_dataset.save_to_disk(tokenized_output_dir)
    print(f"Tokenized data saved to {tokenized_output_dir}")


def train_model(tokenized_data_dir, model_name, output_dir, logging_dir):
    dataset = load_from_disk(tokenized_data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["query_key_value", "dense"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=3,
        num_train_epochs=1,
        logging_dir=logging_dir,
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        save_total_limit=200,
        report_to=["tensorboard"],
        dataloader_num_workers=28
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            LMEvalCallback(
                tokenizer_name=model_name,
                eval_interval=250,
                eval_tasks=["hellaswag", "mmlu", "belebele"],
                output_dir=os.path.join(output_dir, "lm_eval"),
                tb_logdir=logging_dir
            )
        ]
    )
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "adapter"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--model_name", required=True, default="bigscience/bloom-560m")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.tokenized_dir):
        tokenize_data(args.data_dir, args.model_name, args.tokenized_dir)
    else:
        print(f"Tokenized data already exists at {args.tokenized_dir}, skipping tokenization.")

    train_model(args.tokenized_dir, args.model_name, args.output_dir, args.logging_dir)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()