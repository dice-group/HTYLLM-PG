"""
Usage: python -m htyllm_pg.tokenize_data /path/to/sharded_samples /path/to/output tokenizer.json 5000
"""
import sys
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from .util.yield_jsonl_gz import yield_jsonl_gz


def tokenize_and_save(input_folder, output_folder, tokenizer_path="tokenizer.json", batch_size=10000):
    """
    Tokenize all .jsonl.gz files in subfolders and save as numpy arrays.
    """
    tokenizer = Tokenizer.from_file(path=tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, 
        bos_token="<|endoftext|>", 
        eos_token="<|endoftext|>"
    )
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for subfolder in sorted(input_path.iterdir()):
        if not subfolder.is_dir():
            continue
            
        print(f"Processing subfolder: {subfolder.name}")
        subfolder_output = output_path / subfolder.name
        subfolder_output.mkdir(parents=True, exist_ok=True)
        
        file_counter = 0
        all_tokens = []
        
        for batch in tqdm(yield_jsonl_gz(str(subfolder), batch_size=batch_size)):
            encoded = tokenizer(batch, add_special_tokens=True)
            
            for token_ids in encoded['input_ids']:
                all_tokens.extend(token_ids)
                
                if len(all_tokens) >= 100_000_000:
                    output_file = subfolder_output / f"tokens_{file_counter:05d}.npy"
                    np.save(output_file, np.array(all_tokens, dtype=np.uint16))
                    print(f"  Saved {output_file.name} ({len(all_tokens)} tokens)")
                    all_tokens = []
                    file_counter += 1
        
        if all_tokens:
            output_file = subfolder_output / f"tokens_{file_counter:05d}.npy"
            np.save(output_file, np.array(all_tokens, dtype=np.uint16))
            print(f"  Saved {output_file.name} ({len(all_tokens)} tokens)")
        
        print(f"Completed {subfolder.name}\n")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m htyllm_pg.tokenize_data <input_folder> <output_folder> [tokenizer_path] [batch_size]")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    tokenizer_path = sys.argv[3] if len(sys.argv) > 3 else "tokenizer.json"
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    
    tokenize_and_save(input_folder, output_folder, tokenizer_path, batch_size)
