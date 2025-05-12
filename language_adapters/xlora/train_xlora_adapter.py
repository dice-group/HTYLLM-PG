import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, AutoConfig
)
import xlora
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

    # load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    config = AutoConfig.from_pretrained(model_name)

    model.config.use_cache = False
    model = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora.xLoRAConfig(
            hidden_size=config.hidden_size,
            base_model_id=model_name,
            xlora_depth=8,
            device=torch.device("cuda"),
            use_trainable_adapters=True,
            adapters = {
                "adapter_1": "./adapter_1",
                "adapter_2": "./adapter_2",
                "adapter_3": "./adapter_3",
                "adapter_4": "./adapter_4",
                "adapter_5": "./adapter_5",
                "adapter_6": "./adapter_6",
                "adapter_7": "./adapter_7",
                "adapter_8": "./adapter_8",
            }
        ),
        verbose=True
    )
    model.set_topk_lora(1)

    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=28,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_dir=logging_dir,
        logging_steps=50,
        save_strategy="steps",
        save_steps=20,
        bf16=True,
        save_total_limit=20,
        report_to=["wandb"],
        run_name="weights_an_biases_test",
        dataloader_num_workers=24
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[LMEvalCallback(
            tokenizer_name=model_name,
            base_model_path=model_name,
            eval_interval=25,
            eval_tasks=["hellaswag", "mmlu", "belebele"],
            output_dir=os.path.join(output_dir, "lm_eval"),
            tb_logdir=logging_dir
        )]
    )

    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "xlora_adapter"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    train_model(args.tokenized_dir, args.model_name, args.output_dir, args.logging_dir)
