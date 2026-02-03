"""
Usage: python -m htyllm_pg.util.test_tokenizer [tokenizer_path] [text]
"""
import sys
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def test_tokenizer(tokenizer_path="tokenizer.json", text=None):
    text = text or "Быстрая коричневая лиса прыгает через ленивую собаку"

    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, 
        bos_token="<|endoftext|>", 
        eos_token="<|endoftext|>"
    )
    
    print(f"Text: {text}")
    encoded = tokenizer(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"IDs: {encoded['input_ids']}")
    print(f"Decoded: {tokenizer.decode(encoded['input_ids'])}")


if __name__ == '__main__':
    tokenizer_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer.json"
    text = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None
    test_tokenizer(tokenizer_path, text)

