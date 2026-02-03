"""
Usage: python -m htyllm_pg.tokenize_subsets /path/to/samples /path/to/output tokenizer.json five_representatives_mediods 2048 10000
"""
import sys
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from .util.yield_jsonl_gz import yield_jsonl_gz
from . import sampling
from .packing import pack_documents

def tokenize_subsets(input_folder, output_folder, tokenizer_path, subset_name, seq_length=2048, batch_size=10000, pack_batch_size=10000):
    """Tokenize specific language subsets, pack using OBFD, and save as numpy arrays."""
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
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
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
        doc_buffer = []
        total_sequences = 0
        
        for batch in tqdm(yield_jsonl_gz(str(subfolder), batch_size=batch_size), desc=lang_code):
            encoded = tokenizer(batch, add_special_tokens=True)
            
            # Accumulate tokenized documents
            for token_ids in encoded['input_ids']:
                if len(token_ids) > 0:
                    doc_buffer.append(token_ids)
                
                # Pack when buffer is full
                if len(doc_buffer) >= pack_batch_size:
                    sequences, masks = pack_documents(doc_buffer, seq_length, pad_token_id)
                    
                    if len(sequences) > 0:
                        # Save sequences and masks
                        seq_file = subfolder_output / f"tokens_{file_counter:05d}.npy"
                        mask_file = subfolder_output / f"masks_{file_counter:05d}.npy"
                        
                        np.save(seq_file, sequences.astype(np.uint32))
                        np.save(mask_file, masks.astype(np.uint8))
                        
                        total_sequences += len(sequences)
                        print(f"  Saved {seq_file.name} ({len(sequences)} sequences)")
                        file_counter += 1
                    
                    doc_buffer = []
        
        # Pack remaining documents
        if doc_buffer:
            sequences, masks = pack_documents(doc_buffer, seq_length, pad_token_id)
            
            if len(sequences) > 0:
                seq_file = subfolder_output / f"tokens_{file_counter:05d}.npy"
                mask_file = subfolder_output / f"masks_{file_counter:05d}.npy"
                
                np.save(seq_file, sequences.astype(np.uint32))
                np.save(mask_file, masks.astype(np.uint8))
                
                total_sequences += len(sequences)
                print(f"  Saved {seq_file.name} ({len(sequences)} sequences)")
        
        print(f"Completed {lang_code}: {total_sequences} packed sequences\n")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python -m htyllm_pg.tokenize_subsets <input_folder> <output_folder> <tokenizer_path> <subset_name> [seq_length] [batch_size]")
        print("Available subsets: five_representatives_mediods, ten_representatives_mediods, twenty_two_representatives_mediods, etc.")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    tokenizer_path = sys.argv[3]
    subset_name = sys.argv[4]
    seq_length = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    batch_size = int(sys.argv[6]) if len(sys.argv) > 6 else 10000
    
    tokenize_subsets(input_folder, output_folder, tokenizer_path, subset_name, seq_length, batch_size)

