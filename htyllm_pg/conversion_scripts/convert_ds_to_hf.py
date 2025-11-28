import os
import json
import torch
import argparse
import deepspeed
import inspect
import sys

# Ensure we can import from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from htyllm_pg.model_builder import MoE_Transformer, moe_builder

class HTYLLMConfig(PretrainedConfig):
    model_type = "htyllm_moe"
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 262144)
        self.max_seq_len = kwargs.get("max_seq_len", 2048)
        self.dim = kwargs.get("dim", 512)
        self.depth = kwargs.get("depth", 12)
        self.heads = kwargs.get("heads", 12)
        self.mlp_dim = kwargs.get("mlp_dim", 2048)
        self.dim_head = kwargs.get("dim_head", 64)
        self.dropout = kwargs.get("dropout", 0.0)
        self.emb_dropout = kwargs.get("emb_dropout", 0.0)
        self.moe_layers = kwargs.get("moe_layers", [0, 3, 6, 9])
        self.num_experts = kwargs.get("num_experts", 8)
        self.k = kwargs.get("k", -1)
        self.capacity_factor = kwargs.get("capacity_factor", 1.5)
        self.eval_capacity_factor = kwargs.get("eval_capacity_factor", 2.0)
        self.min_capacity = kwargs.get("min_capacity", 0.0)
        self.use_residual = kwargs.get("use_residual", False)
        self.gate_backward = kwargs.get("gate_backward", "ste")
        self.ep_size = kwargs.get("ep_size", 1)
        self.topany_gating_impl = kwargs.get("topany_gating_impl", "sparse")
        self.use_flash_attention = kwargs.get("use_flash_attention", False)
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", True)
        super().__init__(**kwargs)

class HTYLLMForCausalLM(PreTrainedModel):
    config_class = HTYLLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        # Filter config args to match MoE_Transformer signature
        valid_args = inspect.signature(MoE_Transformer.__init__).parameters.keys()
        model_kwargs = {k: v for k, v in config.__dict__.items() if k in valid_args}
        self.model = MoE_Transformer(**model_kwargs)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits, aux_loss = self.model(input_ids, attention_mask)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1)) + 0.01 * aux_loss
        return CausalLMOutputWithPast(loss=loss, logits=logits)

def convert(args):
    # 1. Load Config
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config_data = json.load(f)
            config = HTYLLMConfig(**config_data)
    else:
        config = HTYLLMConfig()
    
    # 2. Instantiate HF Model (creates PyTorch model internally)
    print(f"Initializing model with config: {config}")
    
    # Use builder to match training exactly
    builder_args = inspect.signature(moe_builder).parameters.keys()
    builder_kwargs = {k: v for k, v in config.__dict__.items() if k in builder_args}
    model_pytorch = moe_builder(**builder_kwargs)

    hf_model = HTYLLMForCausalLM(config)
    hf_model.model = model_pytorch
    
    # CRITICAL: Add auto_map for trust_remote_code=True support
    hf_model.config.auto_map = {
        "AutoConfig": "modeling_htyllm.HTYLLMConfig",
        "AutoModelForCausalLM": "modeling_htyllm.HTYLLMForCausalLM"
    }
    
    # 3. Initialize DeepSpeed to load checkpoint
    ds_config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "zero_optimization": {"stage": 0},
    }
    
    base_params = {"params": [p for p in hf_model.model.parameters() if p.requires_grad], "name": "parameters"}
    param_groups = split_params_into_different_moe_groups_for_optimizer(base_params)

    print("Initializing DeepSpeed...")
    model_engine, _, _, _ = deepspeed.initialize(
        model=hf_model.model,
        model_parameters=param_groups,
        config=ds_config,
        args=args
    )

    # 4. Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    parent_dir = os.path.dirname(args.checkpoint_path)
    tag = os.path.basename(args.checkpoint_path)
    
    load_path, _ = model_engine.load_checkpoint(parent_dir, tag=tag)
    if load_path is None:
        load_path, _ = model_engine.load_checkpoint(args.checkpoint_path)
        if load_path is None:
            raise ValueError(f"Could not load checkpoint from {args.checkpoint_path}")
            
    print("Checkpoint loaded successfully.")

    # 5. Save HF Model
    print(f"Saving to {args.output_dir}...")
    hf_model.save_pretrained(args.output_dir)
    
    # We read model_builder.py and append our wrapper classes
    model_builder_path = os.path.join(os.path.dirname(__file__), "../model_builder.py")
    with open(model_builder_path, "r") as f:
        model_code = f.read()
    
    wrapper_code = f"""
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import deepspeed
import inspect

# Initialize DeepSpeed distributed backend if not already initialized
# This is required for the MoE layer to function correctly during inference
if not deepspeed.comm.is_initialized():
    deepspeed.init_distributed(dist_backend="nccl")

{inspect.getsource(HTYLLMConfig)}

{inspect.getsource(HTYLLMForCausalLM)}
"""
    
    with open(os.path.join(args.output_dir, "modeling_htyllm.py"), "w") as f:
        f.write(model_code + "\n" + wrapper_code)

    print(f"Saved modeling_htyllm.py to {args.output_dir}")
    print("Saved.")

    # 6. Verify
    print("Verifying equivalence...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_engine.module.to(device).eval()
    hf_model.to(device).eval()
    
    input_ids = torch.tensor([[0, 1, 2]]).to(device)
    with torch.no_grad():
        logits_ds, _ = model_engine.module(input_ids)
        logits_hf = hf_model(input_ids).logits

    diff = (logits_ds - logits_hf).abs().max().item()
    print(f"Max Logit Difference: {diff}")
    if diff < 1e-5:
        print("SUCCESS: Models match!")
    else:
        print("WARNING: Models do not match exactly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to DeepSpeed checkpoint folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HF model")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    convert(args)