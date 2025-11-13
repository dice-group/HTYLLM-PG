"""
Usage: python -m htyllm_pg.tokenize_subsets /path/to/samples /path/to/output tokenizer.json five_representatives_mediods 10000
"""
import sys
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from .util.yield_jsonl_gz import yield_jsonl_gz
from . import sampling

def tokenize_subsets(input_folder, output_folder, tokenizer_path, subset_name, batch_size=10000):
    """Tokenize specific language subsets and save as numpy arrays."""
    # Get subset list
    subset_list = getattr(sampling.language_subsets, subset_name)
    print(f"Processing {len(subset_list)} languages from subset: {subset_name}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(path=tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, 
        bos_token="<|endoftext|>", 
        eos_token="<|endoftext|>"
    )
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process only specified subfolders
    for lang_code in subset_list:
        subfolder = input_path / lang_code
        if not subfolder.exists():
            print(f"Warning: {lang_code} not found, skipping")
            continue
            
        print(f"Processing: {lang_code}")
        subfolder_output = output_path / lang_code
        subfolder_output.mkdir(parents=True, exist_ok=True)
        
        file_counter = 0
        all_tokens = []
        
        for batch in tqdm(yield_jsonl_gz(str(subfolder), batch_size=batch_size), desc=lang_code):
            encoded = tokenizer(batch, add_special_tokens=True)
            
            for token_ids in encoded['input_ids']:
                all_tokens.extend(token_ids)
                
                if len(all_tokens) >= 100_000_000:
                    output_file = subfolder_output / f"tokens_{file_counter:05d}.npy"
                    np.save(output_file, np.array(all_tokens, dtype=np.uint32))
                    print(f"  Saved {output_file.name} ({len(all_tokens)} tokens)")
                    all_tokens = []
                    file_counter += 1
        
        if all_tokens:
            output_file = subfolder_output / f"tokens_{file_counter:05d}.npy"
            np.save(output_file, np.array(all_tokens, dtype=np.uint32))
            print(f"  Saved {output_file.name} ({len(all_tokens)} tokens)")
        
        print(f"Completed {lang_code}\n")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python -m htyllm_pg.tokenize_subsets <input_folder> <output_folder> <tokenizer_path> <subset_name> [batch_size]")
        print("Available subsets: five_representatives_mediods, ten_representatives_mediods, twenty_two_representatives_mediods, etc.")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    tokenizer_path = sys.argv[3]
    subset_name = sys.argv[4]
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    
    tokenize_subsets(input_folder, output_folder, tokenizer_path, subset_name, batch_size)

