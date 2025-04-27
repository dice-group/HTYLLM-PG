import argparse
import torch
import torch.nn.functional as F
import tiktoken
import os
from model.gpt_2_multi_gpu import GPT, GPTConfig  # Assuming you save your model class code in model.py


def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    
    # strip _orig_mod. prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model



def generate_text(model, prompt, enc, device, max_new_tokens=50, temperature=1.0, top_k=50):
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k)
            next_token = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, next_token)
            tokens = torch.cat([tokens, next_token], dim=1)

    output = enc.decode(tokens[0].tolist())
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on.')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Number of tokens to generate.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling.')
    args = parser.parse_args()

    device = args.device
    enc = tiktoken.get_encoding("gpt2")

    model = load_model(args.checkpoint, device)

    print("\nReady to generate! Type your prompt below. Type 'exit' to quit.")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in {'exit', 'quit'}:
            print("Goodbye!")
            break

        output = generate_text(
            model,
            prompt,
            enc,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"\n=== Generated Text ===\n{output}\n")


if __name__ == '__main__':
    main()
