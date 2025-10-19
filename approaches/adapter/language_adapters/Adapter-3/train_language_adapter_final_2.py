import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ─── CONFIG ────────────────────────────────────────────────────────────────
DEFAULT_TOKENIZED_DIR = "/local/upb/users/h/hpurohit/profiles/unix/cs/HTYLLM - PG/Tokenized_directory/Adapter-4"
DEFAULT_MODEL_NAME    = "bigscience/bloom-560m"
DEFAULT_OUTPUT_DIR    = "/local/upb/users/h/hpurohit/profiles/unix/cs/HTYLLM - PG/Final_Output_directory/Adapter-4-final"
DEFAULT_LOGGING_DIR   = os.path.join(DEFAULT_OUTPUT_DIR, "logs")
BLOCK_SIZE            = 512

# ─── DATA PREPARATION ───────────────────────────────────────────────────────
def group_texts(examples):
    """
    Concatenate and chunk input_ids into fixed-size BLOCK_SIZE pieces,
    and generate attention masks of 1s.
    """
    all_ids = sum(examples["input_ids"], [])
    total_len = (len(all_ids) // BLOCK_SIZE) * BLOCK_SIZE
    chunks = [
        all_ids[i : i + BLOCK_SIZE]
        for i in range(0, total_len, BLOCK_SIZE)
    ]
    return {
        "input_ids": chunks,
        "attention_mask": [[1] * BLOCK_SIZE for _ in chunks],
    }

# ─── TRAINING ──────────────────────────────────────────────────────────────
def train_adapter(
    tokenized_dir: str,
    model_name: str,
    output_dir: str,
    logging_dir: str,
):
    # 1) Load tokenized dataset
    print(f"Loading dataset from {tokenized_dir}")
    ds = load_from_disk(tokenized_dir, keep_in_memory=True)

    # 2) Prepare data: group into fixed-size blocks
    print(f"Grouping into {BLOCK_SIZE}-token blocks …")
    ds = ds.map(
        group_texts,
        batched=True,
        remove_columns=ds.column_names,
    )

    # 3) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 4) Load model on CPU (FP16) for DDP/torchrun to distribute
    print(f"Loading {model_name} in FP16 on CPU")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    # 5) Inject LoRA adapters
    print("Injecting LoRA adapters …")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 6) Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=1e-6,
        warmup_steps=1000,
        logging_dir=logging_dir,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=4,
        report_to=[],
        remove_unused_columns=False,
    )

    # 7) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 8) Start training
    print("Starting training …")
    trainer.train()

    # 9) Save the trained LoRA adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LoRA adapter on BLOOM-560M")
    parser.add_argument("--tokenized_dir", default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--model_name",    default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logging_dir",   default=DEFAULT_LOGGING_DIR)
    args = parser.parse_args()

    # Allow TensorFloat32 on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    train_adapter(
        tokenized_dir=args.tokenized_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )