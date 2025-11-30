import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="hf_models/3_7b_english_step_70000", help="Path to the converted model")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        # Fix for potential tokenizer issues
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "=" * 50)
    print("Interactive Generation Loop")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        
        # Remove token_type_ids if present, as the model doesn't support them
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        try:
            output = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                pad_token_id=tokenizer.pad_token_id
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\nModel: {generated_text}\n")
        except Exception as e:
            print(f"Error during generation: {e}\n")

if __name__ == "__main__":
    main()
