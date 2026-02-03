import os
import torch
import argparse
import deepspeed
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from htyllm_pg.model_builder import moe_builder

def create_dummy(output_dir, config_path=None):
    # Configuration (matching config_3_7b.json or similar)
    vocab_size = 262144
    max_seq_len = 2048
    dim = 512
    depth = 2  # Small depth for speed
    heads = 4
    mlp_dim = 2048
    dim_head = 64
    moe_layers = [0]
    num_experts = 4
    
    # Create model
    model = moe_builder(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
        moe_layers=moe_layers,
        num_experts=num_experts,
        use_flash_attention=False
    )

    # Minimal DeepSpeed config
    ds_config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "zero_optimization": {"stage": 0},
        "fp16": {"enabled": True},
    }

    # Split params
    base_params = {"params": [p for p in model.parameters() if p.requires_grad], "name": "parameters"}
    param_groups = split_params_into_different_moe_groups_for_optimizer(base_params)

    # Initialize
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_config
    )

    # Save checkpoint
    print(f"Saving dummy checkpoint to {output_dir}...")
    # We use a dummy tag "step_100"
    model_engine.save_checkpoint(output_dir, tag="step_100")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="dummy_checkpoints_test", help="Output directory")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    create_dummy(args.output_dir)



