import os
import glob
from lm_harness_eval import LMEvalCallback
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType

# load tokenized data, TODO: integrate into end to end pipeline 
dataset_path = "/data/joel/fineweb2_subset_15_indo_european_tokenized"
dataset = load_from_disk(dataset_path)

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# simple adapter config: TODO: adapt target modules
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# training
training_args = TrainingArguments(
    output_dir="/data/joel/phi2-lora-multilingual",
    per_device_train_batch_size=5,
    gradient_accumulation_steps=5,
    num_train_epochs=1,
    logging_dir="/data/joel/phi2-lora-multilingual/logs",
    logging_steps=50,
    save_strategy="steps",
    save_steps=20,
    fp16=True,
    save_total_limit=2,
    report_to=["tensorboard"],  
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
            eval_interval=21, 
            eval_tasks = [
                "truthfulqa", "mmlu", "hellaswag", "xcopa", "xwinograd",
                "pawsx", "xnli", "lambada", "belebele"
            ],
            #eval_tasks = ["hellaswag"],
            output_dir="/data/joel/phi2-lora-multilingual/lm_eval",
            tb_logdir=training_args.logging_dir,
            )
        ]
)

trainer.train()
model.save_pretrained("/data/joel/phi2-lora-multilingual/adapter")
