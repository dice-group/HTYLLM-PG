import torch
import torch.nn.functional as F
from torch import nn
import os
from torch.utils.data import DataLoader, Dataset
import deepspeed
from deepspeed import comm
import argparse
import wandb
import json
import math
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from htyllm_pg.dataset import create_dataloaders
from tqdm.auto import tqdm


def chunked_cross_entropy(logits, targets, chunk_size=4096, ignore_index=-100):
    """
    Computes CrossEntropyLoss in chunks to avoid OOM from casting 
    huge logit tensors to FP32 all at once.
    """
    # logits: [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
    # targets: [Batch, Seq] -> [Batch*Seq]
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    num_tokens = targets.size(0)
    total_loss = 0.0
    num_valid_tokens = 0  # Counter for non-ignored tokens
    
    for i in range(0, num_tokens, chunk_size):
        end = min(i + chunk_size, num_tokens)
        
        # Slice the tensors 
        chunk_logits = logits[i:end]
        chunk_targets = targets[i:end]
        
        chunk_loss = F.cross_entropy(
            chunk_logits.float(), 
            chunk_targets, 
            reduction='sum', 
            ignore_index=ignore_index
        )
        
        total_loss += chunk_loss
        
        if ignore_index is not None:
            num_valid_tokens += (chunk_targets != ignore_index).sum()
        else:
            num_valid_tokens += (end - i)

    # Average the loss
    return total_loss / num_valid_tokens


def save_config(args, output_dir):
    """Saves the model configuration to a JSON file."""
    config = {
        "vocab_size": args.vocab_size,
        "max_seq_len": args.max_seq_len,
        "n_embd": args.dim,
        "n_layer": args.depth,
        "n_head": args.heads,
        "n_inner": args.mlp_dim,
        "dim_head": args.dim_head,
        "dropout": args.dropout,
        "emb_dropout": args.emb_dropout,
        "use_flash_attention": args.use_flash_attention,
        "use_gradient_checkpointing": args.use_gradient_checkpointing,
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")


def build_tokenizer(tokenizer_path: str):
    """Load tokenizer from tokenizer.json file."""
    if os.path.isdir(tokenizer_path):
        tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Ensure pad token exists (common GPT-2 requirement for batching)
    if tok.eos_token is None and tok.sep_token is not None:
        tok.eos_token = tok.sep_token
    if tok.eos_token is None:
        raise ValueError("Tokenizer must define an eos_token.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    
    return tok


def build_model(vocab_size: int, seq_len: int, dim: int, depth: int, heads: int, 
                mlp_dim: int, dropout: float, use_flash_attention: bool):
    """Build GPT-2 model with specified architecture."""
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=dim,
        n_layer=depth,
        n_head=heads,
        n_inner=mlp_dim,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        use_cache=False,
    )
    
    # Try to enable flash-attn if available and requested
    if use_flash_attention:
        try:
            model = GPT2LMHeadModel(cfg, attn_implementation="flash_attention_2")
        except (TypeError, ValueError):
            # Fallback to default if flash_attention_2 not available
            model = GPT2LMHeadModel(cfg)
    else:
        model = GPT2LMHeadModel(cfg)
    
    return model


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, dest="data_dir", help="Path to tokenized data directory")
    parser.add_argument("--tokenizer-path", type=str, dest="tokenizer_path", default="tokenizer.json", help="Path to tokenizer.json file")
    parser.add_argument("--workers", default=8, type=int, help="Number of workers for the Dataloaders!")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs of the training data!")
    # NOTE: --batch-size is deprecated. Batch size is controlled by DeepSpeed config.
    # This arg is kept for backward compatibility but will be overridden by DeepSpeed config.
    parser.add_argument("--batch-size", default=None, type=int, dest="batch_size", help="[DEPRECATED] Batch size is controlled by DeepSpeed config. This is ignored.")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate for AdamW optimizer!")
    parser.add_argument("--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--checkpoint-dir", type=str, dest="checkpoint_dir", default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-steps", type=int, dest="checkpoint_steps", default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--load-checkpoint", type=int, dest="load_checkpoint", default=None, help="Checkpoint step to load, e.g. 1000")
    
    # Model architecture parameters
    parser.add_argument("--vocab-size", type=int, dest="vocab_size", default=131072, help="Vocabulary size")
    parser.add_argument("--max-seq-len", type=int, dest="max_seq_len", default=2048, help="Maximum sequence length")
    parser.add_argument("--dim", type=int, default=2048, help="Model dimension")
    parser.add_argument("--depth", type=int, default=24, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--mlp-dim", type=int, dest="mlp_dim", default=8192, help="MLP hidden dimension")
    parser.add_argument("--dim-head", type=int, dest="dim_head", default=128, help="Dimension per attention head")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--emb-dropout", type=float, dest="emb_dropout", default=0.0, help="Embedding dropout rate")
    parser.add_argument("--use-flash-attention", action="store_true", dest="use_flash_attention", help="Use Flash Attention 2 if available")
    parser.add_argument("--use-gradient-checkpointing", action="store_true", dest="use_gradient_checkpointing", default=True, help="Use gradient checkpointing to save memory")
    
    parser.add_argument("--train-split", type=float, dest="train_split", default=0.95, help="Fraction of data for training (1.0 = no test split)")
    
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = get_args()
    
    # Load tokenizer
    tokenizer = build_tokenizer(args.tokenizer_path)
    
    # Build model
    model_pytorch = build_model(
        vocab_size=args.vocab_size,
        seq_len=args.max_seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        use_flash_attention=args.use_flash_attention
    )
    
    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        model_pytorch.gradient_checkpointing_enable()
    
    pytorch_total_params = sum(p.numel() for p in model_pytorch.parameters() if p.requires_grad)
    pytorch_trainable_params = sum(p.numel() for p in model_pytorch.parameters() if p.requires_grad)
    
    # Initialize DeepSpeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model_pytorch,
        model_parameters=model_pytorch.parameters(),
        args=args,
    )
    
    RANK = comm.get_rank()
    WORLD_SIZE = comm.get_world_size()
    
    # Extract batch size configuration from DeepSpeed (single source of truth)
    ds_config = model.config
    micro_batch_per_gpu = ds_config.train_micro_batch_size_per_gpu
    grad_accum_steps = ds_config.gradient_accumulation_steps
    global_batch_size = WORLD_SIZE * micro_batch_per_gpu * grad_accum_steps
    tokens_per_step = global_batch_size * args.max_seq_len
    
    # Get optimizer and scheduler config from DeepSpeed config JSON
    # Read the config file directly for reliable access to optimizer/scheduler params
    ds_config_path = getattr(args, 'deepspeed_config', None)
    ds_config_json = None
    opt_config = None
    sched_config = None
    grad_clip = None
    
    if ds_config_path and os.path.exists(ds_config_path):
        with open(ds_config_path, 'r') as f:
            ds_config_json = json.load(f)
            opt_config = ds_config_json.get('optimizer', {})
            sched_config = ds_config_json.get('scheduler', None)
            grad_clip = ds_config_json.get('gradient_clipping', 0)
    
    # Fallback: try to get from model.config if JSON read failed
    if opt_config is None:
        try:
            opt_config = ds_config.optimizer_config if hasattr(ds_config, 'optimizer_config') else {}
        except Exception:
            opt_config = {}
    
    # Fallback for gradient clipping
    if grad_clip is None:
        if hasattr(ds_config, 'gradient_clipping'):
            grad_clip = ds_config.gradient_clipping
        else:
            grad_clip = 0
    
    # Print comprehensive run identity block
    if RANK == 0:
        print("\n" + "="*80)
        print("RUN IDENTITY - GPT-2 Dense Baseline")
        print("="*80)
        print(f"Model Architecture:")
        print(f"  Total Parameters:        {pytorch_total_params:,}")
        print(f"  Trainable Parameters:    {pytorch_trainable_params:,}")
        print(f"  Vocabulary Size:        {args.vocab_size:,}")
        print(f"  Max Sequence Length:    {args.max_seq_len}")
        print(f"  Model Dimension:        {args.dim}")
        print(f"  Number of Layers:       {args.depth}")
        print(f"  Attention Heads:         {args.heads}")
        print(f"  Head Dimension:         {args.dim_head}")
        print(f"  MLP Hidden Dimension:   {args.mlp_dim}")
        print(f"  Dropout:                {args.dropout}")
        print(f"  Flash Attention:        {args.use_flash_attention}")
        print(f"  Gradient Checkpointing:  {args.use_gradient_checkpointing}")
        print()
        print(f"Distributed Training:")
        print(f"  World Size (GPUs):      {WORLD_SIZE}")
        print(f"  Micro Batch per GPU:    {micro_batch_per_gpu}")
        print(f"  Gradient Accum Steps:   {grad_accum_steps}")
        print(f"  Global Batch Size:      {global_batch_size}")
        print(f"  Tokens per Step:        {tokens_per_step:,}")
        print()
        print(f"Optimizer (AdamW):")
        if opt_config and 'params' in opt_config:
            opt_params = opt_config['params']
            print(f"  Learning Rate:         {opt_params.get('lr', 'N/A')}")
            print(f"  Betas:                  {opt_params.get('betas', 'N/A')}")
            print(f"  Epsilon:               {opt_params.get('eps', 'N/A')}")
            print(f"  Weight Decay:           {opt_params.get('weight_decay', 'N/A')}")
        else:
            print(f"  [Config not available]")
        if grad_clip and grad_clip > 0:
            print(f"  Gradient Clipping:     {grad_clip}")
        print()
        if sched_config and 'params' in sched_config:
            sched_params = sched_config['params']
            print(f"Scheduler ({sched_config.get('type', 'WarmupDecayLR')}):")
            print(f"  Warmup Min LR:         {sched_params.get('warmup_min_lr', 'N/A')}")
            print(f"  Warmup Max LR:         {sched_params.get('warmup_max_lr', 'N/A')}")
            print(f"  Warmup Steps:          {sched_params.get('warmup_num_steps', 'N/A')}")
            print(f"  Total Steps:            {sched_params.get('total_num_steps', 'N/A')}")
            print()
        print(f"Training Configuration:")
        print(f"  Epochs:                 {args.epochs}")
        print(f"  Checkpoint Steps:       {args.checkpoint_steps}")
        print(f"  Data Workers:           {args.workers}")
        print(f"  Train Split:            {args.train_split}")
        print("="*80 + "\n")
    
    # Initialize wandb on rank 0
    if RANK == 0:
        # Get wandb key from environment variable (fallback to project default)
        wandb_key = os.getenv("WANDB_API_KEY", "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")
        wandb.login(key=wandb_key)
        wandb.init(
            project="htyllm-pg",
            name="gpt2_dense_baseline_2048d_24l_16h",
            config={
                "vocab_size": args.vocab_size,
                "max_seq_len": args.max_seq_len,
                "dim": args.dim,
                "depth": args.depth,
                "heads": args.heads,
                "mlp_dim": args.mlp_dim,
                "dim_head": args.dim_head,
                "dropout": args.dropout,
                "emb_dropout": args.emb_dropout,
                "use_flash_attention": args.use_flash_attention,
                "use_gradient_checkpointing": args.use_gradient_checkpointing,
                "lr": opt_config.get('params', {}).get('lr', args.lr) if opt_config else args.lr,
                "weight_decay": opt_config.get('params', {}).get('weight_decay', args.weight_decay) if opt_config else args.weight_decay,
                "betas": opt_config.get('params', {}).get('betas', [0.9, 0.999]) if opt_config else [0.9, 0.999],
                "eps": opt_config.get('params', {}).get('eps', 1e-8) if opt_config else 1e-8,
                "micro_batch_per_gpu": micro_batch_per_gpu,
                "gradient_accumulation_steps": grad_accum_steps,
                "global_batch_size": global_batch_size,
                "tokens_per_step": tokens_per_step,
                "world_size": WORLD_SIZE,
                "epochs": args.epochs,
                "total_trainable_params": pytorch_trainable_params,
                "model_type": "gpt2_dense"
            }
        )
        
        # Define metrics to use 'step' as x-axis
        wandb.define_metric("step")
        wandb.define_metric("train_loss", step_metric="step")
        wandb.define_metric("test_loss", step_metric="step")
        wandb.define_metric("test_ppl", step_metric="step")
        
        # Save config.json for conversion scripts
        save_config(args, args.checkpoint_dir)
    
    # Load checkpoint if specified
    global_step = 0
    if args.load_checkpoint:
        tag = f"step_{args.load_checkpoint}"
        _, client_state = model.load_checkpoint(args.checkpoint_dir, tag=tag)
        global_step = int(args.load_checkpoint)
        if RANK == 0:
            print(f"Loaded checkpoint '{tag}', resuming from step {global_step}")
    
    # Use pad_token_id = 0 
    PAD_TOKEN_ID = 0
    # Ensure ignore_index matches dataset (-100)
    if RANK == 0:
        print(f"PAD_TOKEN_ID: {PAD_TOKEN_ID}")
    
    # Load data - use micro_batch_per_gpu from DeepSpeed config (single source of truth)
    if args.data_dir:
        train_dataloader, train_sampler, test_dataloader, test_sampler = create_dataloaders(
            args.data_dir, 
            seq_length=args.max_seq_len, 
            batch_size=micro_batch_per_gpu,  # Use DeepSpeed config value
            num_workers=args.workers,
            train_split=args.train_split
        )
    else:
        train_dataset = DummyTextDataset(vocab_size=args.vocab_size, seq_len=args.max_seq_len, num_samples=8_000)
        test_dataset = DummyTextDataset(vocab_size=args.vocab_size, seq_len=args.max_seq_len, num_samples=2_000)
        train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=args.workers, batch_size=micro_batch_per_gpu)
        test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=args.workers, batch_size=micro_batch_per_gpu)
        train_sampler = None
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = batch['input_ids'].to(device)
            target = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss using chunked cross-entropy for large vocab
            loss = chunked_cross_entropy(logits, target)
            
            if RANK == 0 and global_step == 0:
                print(f"=== FIRST BATCH DIAGNOSTICS ===")
                print(f"loss: {loss.item():.4f}")
                print(f"logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                print(f"logits std: {logits.std().item():.2f}")
                print(f"any NaN in logits: {torch.isnan(logits).any().item()}")
                print(f"any Inf in logits: {torch.isinf(logits).any().item()}")
                print(f"input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                print(f"num tokens with label=-100: {(target == -100).sum().item()} / {target.numel()}")
                print(f"================================")
            
            model.backward(loss)
            model.step()
            
            if RANK == 0:
                log_dict = {"train_loss": loss.item(), "epoch": epoch, "step": global_step}
                wandb.log(log_dict)
            
            # Save checkpoint periodically
            global_step += 1
            if global_step % args.checkpoint_steps == 0:
                if RANK == 0:
                    print(f"Saving checkpoint at step {global_step}...")
                model.save_checkpoint(args.checkpoint_dir, tag=f"step_{global_step}")
    
    # Evaluation after training is complete (only if test data exists)
    if test_dataloader is not None and len(test_dataloader) > 0:
        if RANK == 0:
            print(f"\n{'='*50}")
            print("Training complete! Starting evaluation...")
            print(f"{'='*50}\n")
        
        model.eval()
        with torch.inference_mode():
            test_loss_sum = 0
            num_test_batches = 0
            
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                target = batch['labels'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                test_loss = chunked_cross_entropy(logits, target)
                test_loss_sum += test_loss.item()
                num_test_batches += 1
            
            if num_test_batches > 0:
                avg_test_loss = test_loss_sum / num_test_batches
                test_ppl = math.exp(avg_test_loss)
                
                if RANK == 0:
                    print(f"\n{'='*50}")
                    print(f"Final Test Loss: {avg_test_loss:.4f}")
                    print(f"Final Test Perplexity: {test_ppl:.4f}")
                    print(f"{'='*50}\n")
                    wandb.log({"test_loss": avg_test_loss, "test_ppl": test_ppl})
    else:
        if RANK == 0:
            print(f"\n{'='*50}")
            print("Training complete! (No test split configured)")
            print(f"{'='*50}\n")
    
    # Save final model
    if RANK == 0:
        print("Saving final model...")
        wandb.finish()
    model.save_checkpoint(args.checkpoint_dir, tag="final")
    
    # Save tokenizer
    if RANK == 0:
        tokenizer.save_pretrained(args.checkpoint_dir)


class DummyTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.arange(self.seq_len + 1) % self.vocab_size
        inputs = x[:-1]
        targets = x[1:]

        return {
            'input_ids': inputs,
            'labels': targets
        }


if __name__ == "__main__":
    main()
