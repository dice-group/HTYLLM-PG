#!/usr/bin/env python
"""Debug script to test which strings encode to empty with your tokenizer."""

from transformers import AutoTokenizer

def test_problematic_strings():
    """Test common problematic strings that might encode to empty."""
    
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    
    # Common problematic strings
    test_strings = [
        "\n",           # single newline
        "\n\n",         # double newline  
        "\t",           # tab
        " ",            # single space
        "  ",           # double space
        "\u00A0",       # non-breaking space
        "\u200B",       # zero-width space
        "\u200C",       # zero-width non-joiner
        "\u200D",       # zero-width joiner
        "\u2028",       # line separator
        "\u2029",       # paragraph separator
        "\r",           # carriage return
        "\r\n",         # CRLF
        "",             # empty string
        "\u00AD",       # soft hyphen
        "\uFEFF",       # byte order mark
    ]
    
    print("Testing problematic strings:")
    print("="*50)
    
    for i, test_str in enumerate(test_strings):
        try:
            encoded = tokenizer.encode(test_str, add_special_tokens=False)
            if len(encoded) == 0:
                print(f"❌ EMPTY: {repr(test_str)} -> {encoded}")
            else:
                print(f"✅ OK:    {repr(test_str)} -> {encoded}")
        except Exception as e:
            print(f"💥 ERROR: {repr(test_str)} -> {e}")
    
    print("\n" + "="*50)
    print("Tokenizer info:")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}, PAD={tokenizer.pad_token}, UNK={tokenizer.unk_token}")

if __name__ == "__main__":
    test_problematic_strings() 