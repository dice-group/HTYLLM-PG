#!/usr/bin/env python3
"""
Simple script to train a HuggingFace tokenizer on FineWeb2 multilingual data.
"""

import json
import gzip
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFC
from transformers import PreTrainedTokenizerFast


def get_training_corpus(data_dir, batch_size=1000):
    """Generator that yields batches of text from all gzipped JSONL files."""
    data_path = Path(data_dir)
    batch = []
    
    for lang_dir in sorted(data_path.glob("*.jsonl")):
        if not lang_dir.is_dir():
            continue
        
        print(f"Processing {lang_dir.name}...")
        
        for gz_file in sorted(lang_dir.glob("*.jsonl.gz")):
            with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '').strip()
                        if text:
                            batch.append(text)
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                    except:
                        continue
    
    if batch:
        yield batch


def train_tokenizer(data_dir, vocab_size=32000, output_dir="tokenizer"):
    """Train a BPE tokenizer using HuggingFace tokenizers library."""
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    tokenizer.normalizer = NFC()
    
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2
    )
    
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    training_corpus = get_training_corpus(data_dir)
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Saving tokenizer to {output_dir}/")
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    
    wrapped_tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Tokenizer saved to {output_dir}/")
    print(f"  - tokenizer.json")
    print(f"  - tokenizer_config.json")
    print(f"  - special_tokens_map.json")
    
    return wrapped_tokenizer


def test_tokenizer(tokenizer_dir):
    """Load and test the trained tokenizer."""
    from transformers import AutoTokenizer
    
    print(f"\n=== Testing Tokenizer from {tokenizer_dir} ===")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    test_texts = [
        "Hello world!",
        "བོད་ཡིག་",  
        "ދިވެހި",     
        "नमस्ते",      
        "This is a test.",
    ]
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.all_special_tokens}\n")
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="fineweb2_subset",
                        help="Directory containing language JSONL folders")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size")
    parser.add_argument("--output-dir", type=str, default="tokenizer",
                        help="Output directory for tokenizer")
    parser.add_argument("--test", action="store_true",
                        help="Test tokenizer after training")
    
    args = parser.parse_args()
    
    tokenizer = train_tokenizer(
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir
    )
    
    if args.test:
        test_tokenizer(args.output_dir)