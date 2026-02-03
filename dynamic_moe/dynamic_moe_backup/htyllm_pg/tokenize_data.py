"""
Usage: python -m htyllm_pg.tokenize_data /path/to/sharded_samples /path/to/output tokenizer.json 2048 5000
"""
import sys
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from .util.yield_jsonl_gz import yield_jsonl_gz
from .packing import pack_documents


def tokenize_and_save(input_folder, output_folder, tokenizer_path="tokenizer.json", 
                      seq_length=2048, batch_size=30000, pack_batch_size=30000):
    """
    Tokenize all .jsonl.gz files in subfolders, pack using OBFD, and save as numpy arrays.
    
    Args:
        input_folder: Path to input data
        output_folder: Path to save packed sequences
        tokenizer_path: Path to tokenizer file
        seq_length: Maximum sequence length (L)
        batch_size: Number of documents to process at once
        pack_batch_size: Number of documents to accumulate before packing
    """
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
    
    for subfolder in sorted(input_path.iterdir()):
        if not subfolder.is_dir():
            continue
            
        print(f"Processing subfolder: {subfolder.name}")
        subfolder_output = output_path / subfolder.name
        subfolder_output.mkdir(parents=True, exist_ok=True)
        
        file_counter = 0
        doc_buffer = []
        total_sequences = 0
        
        for batch in tqdm(yield_jsonl_gz(str(subfolder), batch_size=batch_size)):
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
        
        print(f"Completed {subfolder.name}: {total_sequences} packed sequences\n")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m htyllm_pg.tokenize_data <input_folder> <output_folder> [tokenizer_path] [seq_length] [batch_size]")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    tokenizer_path = sys.argv[3] if len(sys.argv) > 3 else "tokenizer.json"
    seq_length = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    
    tokenize_and_save(input_folder, output_folder, tokenizer_path, seq_length, batch_size)

