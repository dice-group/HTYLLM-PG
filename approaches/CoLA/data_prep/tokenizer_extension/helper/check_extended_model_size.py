# validate_both_tiers.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "meta-llama/Llama-3.1-8B"

tier_paths = {
    "tier1": "/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier1/merged_model",
    "tier2": "/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier2/merged_model",
}

base_tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
base_vocab = set(base_tok.get_vocab())

def check_tier(name, path):
    print(f"\n=== {name.upper()} ===")
    ext_tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    print("Base vocab:", len(base_tok))
    print("Extended vocab:", len(ext_tok))

    # new tokens
    ext_vocab = set(ext_tok.get_vocab())
    new_tokens = sorted(ext_vocab - base_vocab)
    print("New tokens:", len(new_tokens))

    # load model + embeddings
    model = AutoModelForCausalLM.from_pretrained(path)
    emb = model.get_input_embeddings().weight
    print("Embedding shape:", tuple(emb.shape))

    # zero-vector test
    new_ids = [ext_tok.convert_tokens_to_ids(t) for t in new_tokens]
    zero = sum(int(torch.all(emb[i] == 0)) for i in new_ids)
    print("Zero-vector new tokens:", zero)

    # cosine test for first token
    if new_tokens:
        tok = new_tokens[0]
        tid = ext_tok.convert_tokens_to_ids(tok)
        text = ext_tok.convert_tokens_to_string([tok])
        base_ids = base_tok(text, add_special_tokens=False)["input_ids"]

        if base_ids:
            base_mean = emb[base_ids].mean(dim=0)
            cos = torch.nn.functional.cosine_similarity(base_mean, emb[tid], dim=0)
            print(f"Cosine(sim) for '{tok}':", float(cos))
        else:
            print("Example token has no base-mapping.")

for name, path in tier_paths.items():
    check_tier(name, path)
