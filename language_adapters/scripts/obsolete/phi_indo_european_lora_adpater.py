import os
import glob
import torch
from lm_harness_eval import LMEvalCallback
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training

torch.backends.cuda.matmul.allow_tf32 = True

# load tokenized data, TODO: integrate into end to end pipeline 
dataset_path = "/data/fineweb2_subset_belebele_tokenized_bloom-560m"
dataset = load_from_disk(dataset_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
    )

model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

# simple adapter config: TODO: adapt target modules
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query_key_value", "dense"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# training
training_args = TrainingArguments(
    output_dir="/data/joel/bloom560m-belebele-languages",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=3,
    num_train_epochs=1,
    logging_dir="/data/joel/bloom560m-belebele-languages/logs",
    logging_steps=20,
    save_strategy="steps",
    save_steps=10,
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
            eval_interval=15, 
            #eval_tasks = [
            #    "truthfulqa", "mmlu", "hellaswag", "xcopa", "xwinograd",
            #    "pawsx", "xnli", "lambada", "belebele"
            #],
            eval_tasks = ["hellaswag", "mmlu", "belebele"],
            output_dir="/data/joel/bloom560m-belebele-languages/lm_eval",
            tb_logdir=training_args.logging_dir,
            )
        ]
)

trainer.train()
model.save_pretrained("/data/joel/bloom560m-belebele-languages/adapter")
