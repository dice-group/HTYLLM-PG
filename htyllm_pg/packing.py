"""
Best-Fit Packing Algorithm for LLM Training Data
Implements OBFD (Optimized Best-Fit-Decreasing) with Segment Tree
"""
import numpy as np
from typing import List, Dict, Tuple


class SegmentTree:
    """Segment tree to track available bin capacities."""
    
    def __init__(self, L: int):
        self.L = L
        self.n = 1 << (L - 1).bit_length()  # next power of 2
        self.tree = np.zeros(2 * self.n, dtype=np.int32)
    
    def insert(self, capacity: int):
        """Mark that a bin with this remaining capacity exists."""
        if capacity <= 0 or capacity > self.L:
            return
        pos = self.n + capacity - 1
        self.tree[pos] = capacity
        while pos > 1:
            pos //= 2
            self.tree[pos] = max(self.tree[2 * pos], self.tree[2 * pos + 1])
    
    def remove(self, capacity: int):
        """Remove a bin with this remaining capacity."""
        if capacity <= 0 or capacity > self.L:
            return
        pos = self.n + capacity - 1
        self.tree[pos] = 0
        while pos > 1:
            pos //= 2
            self.tree[pos] = max(self.tree[2 * pos], self.tree[2 * pos + 1])
    
    def query(self, size: int) -> int:
        """Find smallest available capacity >= size. Returns 0 if none."""
        if self.tree[1] < size:
            return 0
        
        idx = 1
        left = 1
        right = self.n
        
        while left != right:
            mid = (left + right) // 2
            if self.tree[2 * idx] >= size:
                idx = 2 * idx
                right = mid
            else:
                idx = 2 * idx + 1
                left = mid + 1
        
        cap = self.tree[idx]
        return cap if cap >= size else 0


def obfd_packing(items: List[Tuple[int, int]], L: int) -> Dict[int, List[int]]:
    """
    OBFD (Optimized Best-Fit-Decreasing) bin packing.
    
    Args:
        items: List of (item_id, size) tuples
        L: Bin capacity (max sequence length)
    
    Returns:
        Dict mapping bin_id -> [item_ids]
    """
    # Sort by size descending
    items = sorted(items, key=lambda x: x[1], reverse=True)
    
    bin_to_items = {}
    space_to_bins = {}  # remaining_space -> set of bin_ids
    next_bin_id = 0
    seg_tree = SegmentTree(L)
    
    for item_id, size in items:
        if size > L:
            # Skip items that don't fit
            continue
            
        # Find smallest remaining space >= size
        best = seg_tree.query(size)
        
        if best == 0:
            # Create new bin
            bin_id = next_bin_id
            next_bin_id += 1
            
            remaining = L - size
            bin_to_items[bin_id] = [item_id]
            
            if remaining > 0:
                if remaining not in space_to_bins:
                    space_to_bins[remaining] = set()
                space_to_bins[remaining].add(bin_id)
                seg_tree.insert(remaining)
        else:
            # Reuse existing bin
            bin_id = space_to_bins[best].pop()
            
            if not space_to_bins[best]:
                seg_tree.remove(best)
                del space_to_bins[best]
            
            bin_to_items[bin_id].append(item_id)
            new_remaining = best - size
            
            if new_remaining > 0:
                if new_remaining not in space_to_bins:
                    space_to_bins[new_remaining] = set()
                space_to_bins[new_remaining].add(bin_id)
                seg_tree.insert(new_remaining)
    
    return bin_to_items


def pack_documents(docs: List[List[int]], L: int, pad_token_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack tokenized documents into fixed-length sequences using OBFD.
    
    Args:
        docs: List of tokenized documents (list of token IDs)
        L: Maximum sequence length
        pad_token_id: Token ID to use for padding
    
    Returns:
        Tuple of (sequences, masks) as numpy arrays
        - sequences: shape (num_sequences, L) containing token IDs
        - masks: shape (num_sequences, L) where 1=real token, 0=padding
    """
    # Step 1: Build chunks (split long docs)
    chunks = []
    for doc in docs:
        n = len(doc)
        if n == 0:
            continue
        if n <= L:
            chunks.append(doc)
        else:
            # Split long document into chunks
            start = 0
            while start < n:
                end = min(start + L, n)
                chunks.append(doc[start:end])
                start = end
    
    if not chunks:
        return np.array([], dtype=np.int32).reshape(0, L), np.array([], dtype=np.int32).reshape(0, L)
    
    # Step 2: Create items for packing
    items = [(i, len(chunk)) for i, chunk in enumerate(chunks)]
    
    # Step 3: Pack using OBFD
    bin_assignments = obfd_packing(items, L)
    
    # Step 4: Build final sequences with padding
    sequences = []
    masks = []
    
    for bin_id in sorted(bin_assignments.keys()):
        seq = []
        mask = []
        
        for chunk_id in bin_assignments[bin_id]:
            chunk = chunks[chunk_id]
            seq.extend(chunk)
            mask.extend([1] * len(chunk))
        
        # Pad to length L
        if len(seq) < L:
            seq.extend([pad_token_id] * (L - len(seq)))
            mask.extend([0] * (L - len(mask)))
        
        sequences.append(seq)
        masks.append(mask)
    
    return np.array(sequences, dtype=np.int32), np.array(masks, dtype=np.int32)

