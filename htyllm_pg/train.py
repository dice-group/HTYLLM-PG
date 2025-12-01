from numpy import argmax
from torch import nn
import torch
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import deepspeed
from deepspeed import comm
import argparse
import wandb
import wandb
import json
from htyllm_pg.model_builder import moe_builder
from htyllm_pg.dataset import create_dataloaders
from htyllm_pg.util.visualization import create_expert_heatmap
from tqdm.auto import tqdm

from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer


# TODO: LF: naive solution maybe? We should look at this: 
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
    num_valid_tokens = 0 # Counter for non-ignored tokens
    
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
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": args.mlp_dim,
        "dim_head": args.dim_head,
        "dropout": args.dropout,
        "emb_dropout": args.emb_dropout,
        "moe_layers": args.moe_layers,
        "num_experts": args.num_experts,
        "k": args.k,
        "capacity_factor": args.capacity_factor,
        "eval_capacity_factor": args.eval_capacity_factor,
        "min_capacity": args.min_capacity,
        "use_residual": args.use_residual,
        "gate_backward": args.gate_backward,
        "ep_size": args.ep_size,
        "topany_gating_impl": args.topany_gating_impl,
        "use_flash_attention": args.use_flash_attention,
        "use_gradient_checkpointing": args.use_gradient_checkpointing,
        "architectures": ["HTYLLMForCausalLM"],
        "model_type": "htyllm_moe"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")


def get_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, dest="data_dir", help="Path to tokenized data directory")
    parser.add_argument("--workers", default=8, type=int, help="Number of workers for the Dataloaders!")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs of the training data!")
    parser.add_argument("--batch-size", default=224, type=int, dest="batch_size", help="Batch size for training and testing!")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate for AdamW optimizer!")
    parser.add_argument("--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--checkpoint-dir", type=str, dest="checkpoint_dir", default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-steps", type=int, dest="checkpoint_steps", default=100, help="Save checkpoint every N steps")
    parser.add_argument("--load-checkpoint", type=int, dest="load_checkpoint", default=None, help="Checkpoint step to load, e.g. 1000")
    
    # Model architecture parameters
    parser.add_argument("--vocab-size", type=int, dest="vocab_size", default=262144, help="Vocabulary size")
    parser.add_argument("--max-seq-len", type=int, dest="max_seq_len", default=2048, help="Maximum sequence length")
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp-dim", type=int, dest="mlp_dim", default=2048, help="MLP hidden dimension")
    parser.add_argument("--dim-head", type=int, dest="dim_head", default=64, help="Dimension per attention head")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--emb-dropout", type=float, dest="emb_dropout", default=0.0, help="Embedding dropout rate")
    parser.add_argument("--moe-layers", type=int, nargs='+', dest="moe_layers", default=[0, 3, 6, 9], help="Which layers to use MoE")
    parser.add_argument("--num-experts", type=int, dest="num_experts", default=8, help="Number of experts in MoE layers")
    parser.add_argument("--k", type=int, default=-1, help="Top-k gating value")
    parser.add_argument("--capacity-factor", type=float, dest="capacity_factor", default=1.5, help="Capacity factor for training")
    parser.add_argument("--eval-capacity-factor", type=float, dest="eval_capacity_factor", default=2.0, help="Capacity factor for evaluation")
    parser.add_argument("--min-capacity", type=float, dest="min_capacity", default=0.0, help="Minimum capacity for experts")
    parser.add_argument("--use-residual", action="store_true", dest="use_residual", help="Use residual connection in MoE")
    parser.add_argument("--gate-backward", type=str, dest="gate_backward", default="ste", help="Gate backward method")
    parser.add_argument("--ep-size", type=int, dest="ep_size", default=1, help="Expert parallel size")
    parser.add_argument("--topany-gating-impl", type=str, dest="topany_gating_impl", default="sparse", help="Top-any gating implementation: 'opt' or 'sparse'")
    parser.add_argument("--use-flash-attention", action="store_true", dest="use_flash_attention", help="Use Flash Attention (optimized) instead of standard attention")
    parser.add_argument("--use-gradient-checkpointing", action="store_true", dest="use_gradient_checkpointing", default=True, help="Use gradient checkpointing to save memory")
    
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = get_args()

    model_pytorch = moe_builder(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        moe_layers=args.moe_layers,
        num_experts=args.num_experts,
        k=args.k,
        capacity_factor=args.capacity_factor,
        eval_capacity_factor=args.eval_capacity_factor,
        min_capacity=args.min_capacity,
        use_residual=args.use_residual,
        gate_backward=args.gate_backward,
        ep_size=args.ep_size,
        topany_gating_impl=args.topany_gating_impl,
        use_flash_attention=args.use_flash_attention,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )

    pytorch_total_params = sum(p.numel() for p in model_pytorch.parameters() if p.requires_grad)

    # Get model parameters for DeepSpeed
    base_params = {
        "params": [p for p in model_pytorch.parameters() if p.requires_grad],
        "name": "parameters",
    }

    # let DeepSpeed split into MoE / non-MoE param groups
    param_groups = split_params_into_different_moe_groups_for_optimizer(base_params)

    model, optimizer, _, _ = deepspeed.initialize(
        model=model_pytorch,
        model_parameters=param_groups,
        args=args,
    )

    RANK = comm.get_rank()
    # Initialize wandb on rank 0
    if RANK == 0:
        print(f"Total parameters: {pytorch_total_params:_}")
        wandb.login(key="844fd819fc05b9e11ac9814b166ab940a5579dfb")
        wandb.init(
            project="htyllm-pg",
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
                "moe_layers": args.moe_layers,
                "num_experts": args.num_experts,
                "k": args.k,
                "capacity_factor": args.capacity_factor,
                "eval_capacity_factor": args.eval_capacity_factor,
                "min_capacity": args.min_capacity,
                "use_residual": args.use_residual,
                "gate_backward": args.gate_backward,
                "ep_size": args.ep_size,
                "topany_gating_impl": args.topany_gating_impl,
                "use_flash_attention": args.use_flash_attention,
                "use_gradient_checkpointing": args.use_gradient_checkpointing,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "total_trainable_params": pytorch_total_params,
            }
        )
        
        # Define metrics to use 'step' as x-axis
        wandb.define_metric("step")
        wandb.define_metric("train_loss", step_metric="step")
        wandb.define_metric("expert_counts/*", step_metric="step")
        wandb.define_metric("expert_percentage/*", step_metric="step")
        wandb.define_metric("expert_metrics/*", step_metric="step")
        wandb.define_metric("test_loss", step_metric="step")
        
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
    print(f"PAD_TOKEN_ID: {PAD_TOKEN_ID}")
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    
    # real data if data_dir provided otherwise dummy data
    if args.data_dir:
        train_dataloader, train_sampler, test_dataloader, test_sampler = create_dataloaders(
            args.data_dir, 
            seq_length=args.max_seq_len, 
            batch_size=args.batch_size, 
            num_workers=args.workers
        )
    else:
        train_dataset = DummyTextDataset(vocab_size=args.vocab_size, seq_len=args.max_seq_len, num_samples=8_000)
        test_dataset = DummyTextDataset(vocab_size=args.vocab_size, seq_len=args.max_seq_len, num_samples=2_000)
        train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=args.workers, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=args.workers, batch_size=args.batch_size)
        train_sampler = None

    for epoch in range(args.epochs): # normalerweise 1 
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

            output, l_aux, expert_counts = model(input_ids, attention_mask=attention_mask)
            
            ce_loss = chunked_cross_entropy(output, target) 

            loss = ce_loss + 0.01 * l_aux

            model.backward(loss)
            model.step()
            
            if RANK == 0:
                log_dict = {"train_loss": loss.item(), "epoch": epoch, "step": global_step}
                
                # Log expert usage per MoE layer
                for layer_name, exp_counts in expert_counts.items():
                    if exp_counts is not None:
                        exp_counts_cpu = exp_counts.detach().cpu().float()
                        num_experts = exp_counts_cpu.numel()
                        total_tokens = exp_counts_cpu.sum().item()
                        
                        # Log individual expert counts
                        for expert_idx in range(num_experts):
                            count = exp_counts_cpu[expert_idx].item()
                            log_dict[f"expert_counts/{layer_name}/expert_{expert_idx}"] = count
                            # Also log as percentage
                            if total_tokens > 0:
                                log_dict[f"expert_percentage/{layer_name}/expert_{expert_idx}"] = (count / total_tokens) * 100
                        
                        # Log expert load balance metrics
                        if total_tokens > 0:
                            # Ideal uniform distribution
                            ideal_per_expert = total_tokens / num_experts
                            # Load imbalance: max/mean ratio (1.0 = perfect balance)
                            mean_count = exp_counts_cpu.mean().item()
                            max_count = exp_counts_cpu.max().item()
                            if mean_count > 0:
                                log_dict[f"expert_metrics/{layer_name}/load_imbalance"] = max_count / mean_count
                            # Coefficient of variation (lower = more balanced)
                            std_count = exp_counts_cpu.std().item()
                            if mean_count > 0:
                                log_dict[f"expert_metrics/{layer_name}/cv"] = std_count / mean_count
                            
                            # Average experts per token
                            # We use input_ids and attention_mask from the outer loop scope
                            num_valid_tokens = input_ids.numel()
                            if attention_mask is not None:
                                num_valid_tokens = attention_mask.sum().item()
                                
                            if num_valid_tokens > 0:
                                log_dict[f"expert_metrics/{layer_name}/avg_experts_per_token"] = total_tokens / num_valid_tokens
                
                # Log heatmap every 100 steps
                if global_step % 100 == 0:
                    heatmap_buf = create_expert_heatmap(expert_counts)
                    if heatmap_buf:
                        log_dict["expert_heatmap"] = wandb.Image(heatmap_buf, caption=f"Expert Usage Step {global_step}")

                wandb.log(log_dict)
            
            # Save checkpoint periodically
            global_step += 1
            if global_step % args.checkpoint_steps == 0:
                if RANK == 0:
                    print(f"Saving checkpoint at step {global_step}...")
                model.save_checkpoint(args.checkpoint_dir, tag=f"step_{global_step}")

    # Evaluation after training is complete
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

            output, l_aux, _ = model(input_ids, attention_mask=attention_mask)
            test_loss = criterion(output.float().transpose(1,2), target) + 0.01 * l_aux
            test_loss_sum += test_loss.item()
            num_test_batches += 1
            
        avg_test_loss = test_loss_sum / num_test_batches
        
        if RANK == 0:
            print(f"\n{'='*50}")
            print(f"Final Test Loss: {avg_test_loss:.4f}")
            print(f"{'='*50}\n")
            wandb.log({"test_loss": avg_test_loss})
            # Test prediction 
            test_pred, _, _ = model(torch.arange(10).unsqueeze(0).to(device))
            print(f"Test prediction shape: {test_pred.shape}")
            print(f"Prediction for [0,...,9]: {torch.argmax(test_pred.squeeze()[9])}")
            print(f"{'='*50}\n")
    
    # Save final model
    if RANK == 0:
        print("Saving final model...")
        wandb.finish()
    model.save_checkpoint(args.checkpoint_dir, tag="final")

class DummyTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.arange(self.seq_len + 1)
        inputs = x[:-1]
        targets = x[1:]

        return {
            'input_ids': inputs,
            'labels': targets
        }


if __name__ == "__main__":
    main()