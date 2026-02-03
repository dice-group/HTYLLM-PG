import torch
import json
import os
import argparse
from transformers import MixtralConfig, MixtralForCausalLM

def load_megatron_checkpoint(checkpoint_path):
    """
    Loads the Megatron-LM mcore checkpoint.
    
    WARNING: This uses weights_only=False, which can execute arbitrary code.
             Only use this with checkpoints from a trusted source.
    """
    print(f"Loading Megatron checkpoint from: {checkpoint_path}")
    # Force loading on CPU to avoid OOM on consumer GPUs.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("Checkpoint loaded successfully.")
    return checkpoint

def create_mixtral_config(megatron_args):
    """
    Creates a Hugging Face MixtralConfig object from Megatron args.
    """
    config = MixtralConfig(
        vocab_size=megatron_args.padded_vocab_size,
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        num_key_value_heads=megatron_args.num_query_groups,
        hidden_act="silu",  # swiglu uses silu
        max_position_embeddings=megatron_args.max_position_embeddings,
        initializer_range=megatron_args.init_method_std,
        rms_norm_eps=megatron_args.norm_epsilon,
        use_cache=True,
        rope_theta=megatron_args.rotary_base,
        tie_word_embeddings=not megatron_args.untie_embeddings_and_output_weights,
        
        # MoE specific parameters
        num_local_experts=megatron_args.num_experts,
        num_experts_per_tok=megatron_args.moe_router_topk,
        router_aux_loss_coef=megatron_args.moe_aux_loss_coeff,
    )
    print("Hugging Face MixtralConfig created:")
    print(config)
    return config

def convert_state_dict(megatron_state_dict, config):
    """
    Converts the Megatron state_dict to the Hugging Face Mixtral format.
    """
    hf_state_dict = {}
    
    # 1. Word Embeddings
    hf_state_dict['model.embed_tokens.weight'] = megatron_state_dict['embedding.word_embeddings.weight']

    # 2. Final LayerNorm
    hf_state_dict['model.norm.weight'] = megatron_state_dict['decoder.final_layernorm.weight']

    # 3. Output LM Head
    hf_state_dict['lm_head.weight'] = megatron_state_dict['output_layer.weight']
    
    # Extract dimensions for splitting weights
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    
    for layer_i in range(config.num_hidden_layers):
        print(f"Converting layer {layer_i}...")
        
        # --- ATTENTION BLOCKS ---
        
        # Split the fused QKV weights
        qkv_weight = megatron_state_dict[f'decoder.layers.{layer_i}.self_attention.linear_qkv.weight']
        
        # Q weights
        q_weight = qkv_weight[:num_heads * head_dim, :]
        hf_state_dict[f'model.layers.{layer_i}.self_attn.q_proj.weight'] = q_weight
        
        # K weights
        k_weight = qkv_weight[num_heads * head_dim : num_heads * head_dim + num_kv_heads * head_dim, :]
        hf_state_dict[f'model.layers.{layer_i}.self_attn.k_proj.weight'] = k_weight
        
        # V weights
        v_weight = qkv_weight[num_heads * head_dim + num_kv_heads * head_dim:, :]
        hf_state_dict[f'model.layers.{layer_i}.self_attn.v_proj.weight'] = v_weight

        # Output projection
        hf_state_dict[f'model.layers.{layer_i}.self_attn.o_proj.weight'] = megatron_state_dict[f'decoder.layers.{layer_i}.self_attention.linear_proj.weight']

        # Pre-Attention LayerNorm
        hf_state_dict[f'model.layers.{layer_i}.input_layernorm.weight'] = megatron_state_dict[f'decoder.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight']
        
        # Post-Attention LayerNorm (which is Pre-MLP in Megatron)
        hf_state_dict[f'model.layers.{layer_i}.post_attention_layernorm.weight'] = megatron_state_dict[f'decoder.layers.{layer_i}.pre_mlp_layernorm.weight']
        
        # --- MoE BLOCKS ---
        
        # Router
        hf_state_dict[f'model.layers.{layer_i}.block_sparse_moe.gate.weight'] = megatron_state_dict[f'decoder.layers.{layer_i}.mlp.router.weight']

        # Experts
        for expert_i in range(config.num_local_experts):
            # Fused FC1 (gate and up projections for SwiGLU)
            fc1_fused = megatron_state_dict[f'decoder.layers.{layer_i}.mlp.experts.linear_fc1.weight{expert_i}']
            
            # Split into gate (w1) and up (w3)
            gate_proj, up_proj = torch.chunk(fc1_fused, 2, dim=0)
            
            hf_state_dict[f'model.layers.{layer_i}.block_sparse_moe.experts.{expert_i}.w1.weight'] = gate_proj
            hf_state_dict[f'model.layers.{layer_i}.block_sparse_moe.experts.{expert_i}.w3.weight'] = up_proj
            
            # FC2 (down projection)
            hf_state_dict[f'model.layers.{layer_i}.block_sparse_moe.experts.{expert_i}.w2.weight'] = megatron_state_dict[f'decoder.layers.{layer_i}.mlp.experts.linear_fc2.weight{expert_i}']
            
    print("State dictionary conversion complete.")
    return hf_state_dict

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Megatron-LM mcore MoE checkpoint to a Hugging Face Mixtral model."
    )
    parser.add_argument(
        "--megatron_checkpoint_path",
        type=str,
        required=True,
        help="Path to the Megatron model checkpoint file (e.g., model_optim_rng.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the converted Hugging Face model will be saved.",
    )
    args = parser.parse_args()
    
    # 1. Load the Megatron checkpoint
    checkpoint = load_megatron_checkpoint(args.megatron_checkpoint_path)
    megatron_args = checkpoint['args']
    
    # Sanity check for tensor parallelism
    if megatron_args.tensor_model_parallel_size > 1:
        print("ERROR: This script is designed for TP=1. You have a tensor parallel model.")
        print("You will need to merge the checkpoints from all tensor parallel ranks first.")
        return
        
    # 2. Create the Hugging Face config
    hf_config = create_mixtral_config(megatron_args)
    
    # 3. Convert the state dictionary
    hf_state_dict = convert_state_dict(checkpoint['model'], hf_config)

    # 4. Create and load the Hugging Face model
    print("Creating Hugging Face Mixtral model...")
    hf_model = MixtralForCausalLM(hf_config)
    hf_model.load_state_dict(hf_state_dict)
    hf_model.to(torch.bfloat16) # Convert model to the correct dtype
    print("Model loaded with converted weights.")

    # 5. Save the model
    print(f"Saving converted model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    hf_model.save_pretrained(args.output_dir)
    print("Model saved successfully.")
    
    # 6. Save the tokenizer config (important for loading later)
    # The tokenizer files (vocab.json, merges.txt) should be copied manually.
    tokenizer_config = {
        "model_max_length": hf_config.max_position_embeddings,
        "padding_side": "left",
        "tokenizer_class": "GPT2Tokenizer"
    }
    with open(os.path.join(args.output_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    print("\nConversion finished!")
    print(f"The Hugging Face model is saved in: {args.output_dir}")
    print("IMPORTANT: You must manually copy your tokenizer files ('vocab.json' and 'merges.txt') into this directory to have a complete, loadable model.")


if __name__ == '__main__':
    main()