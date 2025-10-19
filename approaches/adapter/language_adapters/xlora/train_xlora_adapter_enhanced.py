import argparse
import os
import torch
import wandb
import xlora

from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, AutoConfig
)
from lm_harness_eval import LMEvalCallback, LMHarnessEarlyStoppingCallback

def train_model(args):
    # init wandb, load data and tokenizer
    wandb.init(project=args.eval_wandb_project, name=args.run_name, resume="allow")
    dataset = load_from_disk(args.tokenized_dir, keep_in_memory=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # config for quantizing model
    bnb_config = BitsAndBytesConfig(
       load_in_4bit=True, 
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16
    )
    
    # main model config
    print(bnb_config)
    print(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="balanced",
        torch_dtype=torch.float32,
    )
    config = AutoConfig.from_pretrained(args.model_name)

    # xlora config
    model.config.use_cache = False
    model = xlora.add_xlora_to_model(
        model=model,
        xlora_config=xlora.xLoRAConfig(
                hidden_size=config.hidden_size,
                base_model_id=args.model_name,
                xlora_depth=8,
                device=torch.device("cuda"),
                use_trainable_adapters=True,
                adapters = {
                    "adapter_1": "./mistral_best_adpater_checkpoints/south_asian_2000_best_checkpoint",
                    "adapter_2": "./mistral_best_adpater_checkpoints/swh_Latn_sna_Latn_nya_Latn_2500_best_checkpoint",
                }
            ),
            verbose=True
    )
    model.set_topk_lora(1)
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        bf16=args.bf16,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to.split(","),
        run_name=args.run_name,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,        
    )

    # Just dummy dataset which enables activation of LMHarnessEarlyStoppingCallback
    eval_dataset = dataset.select(range(10)) 
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,  
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            LMEvalCallback(
                tokenizer_name=args.model_name,
                eval_interval=args.eval_interval,
                eval_tasks=args.eval_tasks.split(","),
                output_dir=os.path.join(args.output_dir, "lm_eval"),
                tb_logdir=args.logging_dir,
                batch_size=args.eval_batch_size,
                limit=args.eval_limit,
                cuda_devices=args.eval_cuda_devices,
                wandb_project=args.eval_wandb_project,
                wandb_run_id=wandb.run.id,
            ),
            LMHarnessEarlyStoppingCallback(
                eval_dir=os.path.join(args.output_dir, "lm_eval"),
                metric_names=args.eval_metric_names.split(","),
                patience=args.early_stopping_patience
            ),
            #FlopsProfilerCallback()
        ]
    )

    resume = args.resume_from_checkpoint == "True"
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(os.path.join(args.output_dir, "xlora_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "xlora_adapter"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", required=True)
    
    # Training
    parser.add_argument("--train_batch_size", type=int, default=52)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--save_total_limit", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="wandb,tensorboard")
    parser.add_argument("--run_name", type=str, default="weights_and_biases_test")
    parser.add_argument("--dataloader_num_workers", type=int, default=28)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_accuracy")
    parser.add_argument("--greater_is_better", type=bool, default=True)

    # Eval and early stopping
    parser.add_argument("--eval_interval", type=int, default=1001)
    parser.add_argument("--eval_tasks", type=str, default="belebele")
    parser.add_argument("--eval_metric_names", type=str, default="belebele", help="Comma-separated list of eval tasks used for early stopping scoring")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", choices=["True", "False"], default="False")
    
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Batch size for LM evaluation")
    parser.add_argument("--eval_limit", type=int, default=100, help="Number of examples per eval task")
    parser.add_argument("--eval_cuda_devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES for LM evaluation subprocess")
    parser.add_argument("--eval_log_samples", action="store_true", help="Whether to log eval samples")
    parser.add_argument("--eval_wandb_project", type=str, default="lm_eval_project", help="WandB project name for eval logs")


    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    train_model(args)