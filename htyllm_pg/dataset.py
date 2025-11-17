import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path

class MultiLangTokenDataset(Dataset): 
    def __init__(self, root_dir, seq_length=2048):
        self.root_dir = Path(root_dir)
        self.seq_length = seq_length
        self.files = []
        self.cumulative_sizes = [0]
        self._memmaps = {}
        self._mask_memmaps = {}
        
        # Collect all .npy files from all language directories
        for lang_dir in sorted(self.root_dir.iterdir()):
            if lang_dir.is_dir():
                npy_files = sorted(lang_dir.glob("tokens_*.npy"))
                for f in npy_files:
                    # Check if corresponding mask file exists
                    mask_file = f.parent / f.name.replace("tokens_", "masks_")
                    if not mask_file.exists():
                        print(f"Warning: No mask file found for {f.name}, skipping")
                        continue
                    
                    # Load to get size - expects pre-packed sequences (N, seq_length)
                    arr = np.load(f, mmap_mode='r')
                    if arr.ndim != 2:
                        raise ValueError(
                            f"Expected 2D array (N, seq_length) in {f.name}, got shape {arr.shape}. "
                            "Please use the new tokenization pipeline with packing."
                        )
                    
                    if arr.shape[1] != seq_length:
                        raise ValueError(
                            f"Sequence length mismatch in {f.name}: expected {seq_length}, got {arr.shape[1]}"
                        )
                    
                    n_seqs = len(arr)
                    if n_seqs > 0:
                        self.files.append((f, mask_file))
                        self.cumulative_sizes.append(
                            self.cumulative_sizes[-1] + n_seqs
                        )
        
        self.total_sequences = self.cumulative_sizes[-1]
        print(f"Loaded {len(self.files)} files, {self.total_sequences} sequences")
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Binary search to find which file contains this sequence
        file_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[file_idx]
        
        # Load the files and extract sequence
        filepath, maskpath = self.files[file_idx]
        
        # Cache memmaps
        if filepath not in self._memmaps:
            self._memmaps[filepath] = np.load(filepath, mmap_mode='r')
            self._mask_memmaps[filepath] = np.load(maskpath, mmap_mode='r')
        
        tokens = self._memmaps[filepath]
        masks = self._mask_memmaps[filepath]
        
        # Extract pre-packed sequence
        seq = tokens[local_idx]
        mask = masks[local_idx]
        
        return {
            'input_ids': seq[:-1].astype(np.int64),
            'labels': seq[1:].astype(np.int64),
            'attention_mask': mask[:-1].astype(np.int64)
        }


# Usage with DeepSpeed
def create_dataloaders(data_dir, seq_length=2048, batch_size=8, num_workers=4, 
                       train_split=0.95, seed=42):
    """
    Create train/test DataLoaders with DistributedSampler.
    """
    from torch.utils.data import Subset
    
    full_dataset = MultiLangTokenDataset(data_dir, seq_length)
    
    # Create train/test split
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    
    np.random.seed(seed)
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True,
        seed=seed
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        shuffle=False,  
        drop_last=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, train_sampler, test_loader, test_sampler