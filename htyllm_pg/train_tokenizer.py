import argparse
from htyllm_pg.util.yield_tokens import yield_tokens
from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer, processors, decoders

def train_tokenizer(folder_path, vocab_size=262_144, batch_size=1000):
    tokenizer = Tokenizer(models.BPE())
    
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens="<|endoftext|>")
    
    print(f"Training tokenizer on data from: {folder_path}")
    tokenizer.train_from_iterator(
        iterator=yield_tokens(folder_path, batch_size=batch_size),
        trainer=trainer
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    test_str = "Test the tokenizer roundtrip: Hello, 世界!"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Roundtrip test - Original: {test_str}")
    print(f"Roundtrip test - Decoded:  {decoded}")
    assert test_str == decoded, "Encode-decode roundtrip failed!"
    
    tokenizer.save("tokenizer.json")

def test_tokenizer(tokenizer):
    test_sentences = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "German": "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Russian": "Быстрая коричневая лиса прыгает через ленивую собаку."
    }
    
    print("\n" + "="*60)
    print("TOKENIZER TEST RESULTS")
    print("="*60)
    
    for language, sentence in test_sentences.items():
        encoded = tokenizer.encode(sentence)
        print(f"\n{language}:")
        print(f"  Input:  {sentence}")
        print(f"  Tokens: {encoded.tokens}")
        print(f"  IDs:    {encoded.ids}")
        print(f"  Decoded: {tokenizer.decode(encoded.ids)}")

def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer and test on multiple languages")
    parser.add_argument("folder_path", type=str, help="Path to folder containing training data")
    parser.add_argument("--vocab_size", type=int, default=262_144, help="Vocabulary size (default: 262144)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training (default: 1000)")
    
    args = parser.parse_args()
    
    tokenizer = train_tokenizer(args.folder_path, args.vocab_size, args.batch_size)
    test_tokenizer(tokenizer)
    
    print("\n" + "="*60)
    print("Training and testing completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()