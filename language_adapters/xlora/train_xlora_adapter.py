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
    dataset = dataset.shuffle(seed=42)
    
    #sample_frac = 0.001
    #dataset = dataset.select(range(max(1, int(len(dataset) * sample_frac))))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #torch.autograd.set_detect_anomaly(True)


    # bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True, 
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.float16
    # )
    bnb_config=None

    # load base model with quantization
    print(bnb_config)
    print(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
    )
    print("Model device Map:", model.hf_device_map)
    config = AutoConfig.from_pretrained(model_name)

    model.config.use_cache = False
    model = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora.xLoRAConfig(
            hidden_size=config.hidden_size,
            base_model_id=model_name,
            xlora_depth=1,
            device=torch.device("cuda"),
            use_trainable_adapters=False,
            adapters = {
                "adapter_1": "/upb/users/j/joeldag/profiles/unix/cs/HTYLLM-PG/language_adapters/xlora/mistral_best_adpater_checkpoints/jav_Latn_sun_Latn_500_best_checkpoint",
                "adapter_2": "/upb/users/j/joeldag/profiles/unix/cs/HTYLLM-PG/language_adapters/xlora/mistral_best_adpater_checkpoints/swh_Latn_sna_Latn_nya_Latn_2500_best_checkpoint",
            }
        ),
        verbose=True
    )
    model.set_topk_lora(1)

    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=26,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_dir=logging_dir,
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        save_total_limit=200,
        report_to=["wandb"],
        run_name="weights_an_biases_test",
        dataloader_num_workers=16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        # callbacks=[LMEvalCallback(
        #     tokenizer_name=model_name,
        #     base_model_path=model_name,
        #     eval_interval=25,
        #     eval_tasks=["hellaswag", "mmlu", "belebele"],
        #     output_dir=os.path.join(output_dir, "lm_eval"),
        #     tb_logdir=logging_dir
        # )]
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