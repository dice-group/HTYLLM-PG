#!/usr/bin/env python3
"""Interactive inference for models trained on dummy sequential data."""

import torch
import deepspeed
import argparse
from htyllm_pg.model_builder import moe_builder


def get_args():
    parser = argparse.ArgumentParser(description="Interactive inference for dummy data model")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--tag", type=str, default="final", help="Checkpoint tag (e.g., 'final' or 'step_1000')")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default architecture matching train.py defaults
    model_pytorch = moe_builder(
        vocab_size=10000, # TODO: LF: maybe we need some hyperparameters file saved during trainig, idk 
        max_seq_len=1200,
        dim=512,
        depth=12,
        heads=12,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        moe_layers=[0, 3, 6, 9],
        num_experts=8,
        k=-1,
        capacity_factor=1.5,
        eval_capacity_factor=2.0,
        min_capacity=0.0,
        use_residual=False,
        gate_backward="ste",
        ep_size=1,
        topany_gating_impl="opt_mem",
        use_flash_attention=False
    )
    
    # Initialize with DeepSpeed
    model, _, _, _ = deepspeed.initialize(model=model_pytorch, args=args)
    
    # Load checkpoint
    print(f"Loading checkpoint '{args.tag}' from {args.checkpoint_dir}...")
    model.load_checkpoint(args.checkpoint_dir, tag=args.tag)
    model.eval()
    print("Model loaded successfully!\n")
    
    print("=" * 60)
    print("Interactive Inference Session (Dummy Data Model)")
    print("=" * 60)
    print("Enter space-separated token IDs (e.g., '0 1 2 3')")
    print("Type 'quit' or 'exit' to end session\n")
    
    with torch.inference_mode():
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Parse input as token IDs
                token_ids = [int(x) for x in user_input.split()]
                
                if not token_ids:
                    print("Please enter at least one token ID\n")
                    continue
                
                # Create tensor and predict
                tokens = torch.tensor([token_ids]).to(device)
                output, _ = model(tokens)
                
                # Get prediction for next token
                next_token_logits = output[0, -1, :]
                predicted_token = torch.argmax(next_token_logits).item()
                
                print(f"Input sequence:  {token_ids}")
                print(f"Predicted next:  {predicted_token}\n")
                
            except ValueError:
                print("Error: Please enter valid integer token IDs\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

