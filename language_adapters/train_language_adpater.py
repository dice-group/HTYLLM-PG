import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from lm_harness_eval import LMEvalCallback

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
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=3,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
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
        callbacks=[LMEvalCallback(
            tokenizer_name=model_name,
            eval_interval=250,
            eval_tasks=["hellaswag", "mmlu", "belebele"],
            output_dir=os.path.join(output_dir, "lm_eval"),
            tb_logdir=logging_dir
        )]
    )
    trainer.train()
    #trainer.train(resume_from_checkpoint=True)
    model.save_pretrained(os.path.join(output_dir, "adapter"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_model(args.tokenized_dir, args.model_name, args.output_dir, args.logging_dir)
