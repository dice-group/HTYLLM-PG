"""
Comprehensive tests for the Best-Fit Packing pipeline
"""
import pytest
import numpy as np
import tempfile
import shutil
import gzip
import json
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from htyllm_pg.packing import SegmentTree, obfd_packing, pack_documents
from htyllm_pg.dataset import MultiLangTokenDataset
from htyllm_pg.tokenize_data import tokenize_and_save


class TestSegmentTree:
    """Test segment tree operations"""
    
    def test_basic_operations(self):
        tree = SegmentTree(16)
        
        tree.insert(10)
        tree.insert(5)
        tree.insert(8)
        
        assert tree.query(6) == 8, "Should find smallest capacity >= 6"
        assert tree.query(5) == 5, "Should find exact match"
        assert tree.query(11) == 0, "Should return 0 when no capacity fits"
        
        tree.remove(8)
        assert tree.query(6) == 10, "After removing 8, next is 10"
    
    def test_boundary_cases(self):
        tree = SegmentTree(10)
        
        tree.insert(10)
        assert tree.query(10) == 10, "Should handle max capacity"
        assert tree.query(11) == 0, "Should reject oversized queries"
        
        tree.insert(1)
        assert tree.query(1) == 1, "Should handle min capacity"


class TestOBFDPacking:
    """Test OBFD packing algorithm"""
    
    def test_simple_packing(self):
        L = 10
        items = [(0, 7), (1, 5), (2, 4), (3, 3), (4, 2)]
        
        bins = obfd_packing(items, L)
        
        # Verify all items packed
        all_items = [item for items in bins.values() for item in items]
        assert sorted(all_items) == [0, 1, 2, 3, 4]
        
        # Verify no bin exceeds capacity
        item_sizes = {i: s for i, s in items}
        for bin_items in bins.values():
            total = sum(item_sizes[i] for i in bin_items)
            assert total <= L, f"Bin exceeds capacity: {total} > {L}"
    
    def test_perfect_fit(self):
        L = 10
        items = [(0, 5), (1, 5)]
        
        bins = obfd_packing(items, L)
        
        # Should pack both in one bin (perfect fit)
        assert len(bins) == 1, "Should use only 1 bin for perfect fit"
        assert len(bins[0]) == 2, "Both items should be in same bin"
    
    def test_oversized_items(self):
        L = 10
        items = [(0, 15), (1, 5), (2, 12)]
        
        bins = obfd_packing(items, L)
        
        # Only item 1 should be packed
        all_items = [item for items in bins.values() for item in items]
        assert all_items == [1], "Only items <= L should be packed"


class TestPackDocuments:
    """Test document packing functionality"""
    
    def test_basic_packing(self):
        docs = [
            [1, 2, 3, 4, 5],
            [10, 11, 12],
            [20, 21],
        ]
        L = 10
        PAD = 0
        
        sequences, masks = pack_documents(docs, L, PAD)
        
        # Verify shapes
        assert sequences.shape[1] == L
        assert masks.shape[1] == L
        assert sequences.shape == masks.shape
        
        # Verify all tokens preserved
        total_input = sum(len(d) for d in docs)
        total_real = (masks == 1).sum()
        assert total_input == total_real, "All tokens should be preserved"
        
        # Verify padding correctness
        for seq, mask in zip(sequences, masks):
            real_count = (mask == 1).sum()
            pad_count = (mask == 0).sum()
            
            # All padding should be at the end and be PAD token
            if pad_count > 0:
                assert np.all(seq[-pad_count:] == PAD), "Padding should be PAD token"
                assert np.all(mask[-pad_count:] == 0), "Padding mask should be 0"
    
    def test_long_document_splitting(self):
        L = 10
        long_doc = list(range(25))  # 25 tokens
        
        sequences, masks = pack_documents([long_doc], L, 0)
        
        # Should split into chunks
        total_real = (masks == 1).sum()
        assert total_real == 25, "All 25 tokens should be preserved"
    
    def test_empty_documents(self):
        docs = [[], [1, 2], []]
        L = 10
        
        sequences, masks = pack_documents(docs, L, 0)
        
        # Should skip empty docs
        total_real = (masks == 1).sum()
        assert total_real == 2, "Only non-empty tokens should be packed"
    
    def test_utilization(self):
        # Create docs that should pack efficiently
        docs = [[i] * 5 for i in range(10)]  # 10 docs of 5 tokens each
        L = 10
        
        sequences, masks = pack_documents(docs, L, 0)
        
        # Should pack 2 docs per sequence (50 tokens total, 5 sequences)
        assert len(sequences) == 5, "Should create 5 sequences"
        
        total_real = (masks == 1).sum()
        utilization = total_real / (len(sequences) * L)
        assert utilization == 1.0, "Should have perfect utilization"


class TestEndToEndPipeline:
    """Test the complete pipeline from tokenization to data loading"""
    
    def test_full_pipeline(self):
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        try:
            data_dir = Path(temp_dir) / "data"
            lang_dir = data_dir / "test_lang"
            lang_dir.mkdir(parents=True)
            
            # Simulate tokenized and packed data
            L = 128
            PAD = 0
            
            # Create sample documents
            docs = [
                list(range(1, 50)),      # 49 tokens
                list(range(100, 180)),   # 80 tokens
                list(range(200, 220)),   # 20 tokens
                list(range(300, 450)),   # 150 tokens (will split)
            ]
            
            # Pack documents
            sequences, masks = pack_documents(docs, L, PAD)
            
            # Save packed data
            np.save(lang_dir / "tokens_00000.npy", sequences.astype(np.uint32))
            np.save(lang_dir / "masks_00000.npy", masks.astype(np.uint8))
            
            # Load with dataset
            dataset = MultiLangTokenDataset(data_dir, seq_length=L)
            
            # Verify dataset properties
            assert len(dataset) == len(sequences), "Dataset size should match sequences"
            
            # Test data loading
            for idx in range(len(dataset)):
                batch = dataset[idx]
                
                # Verify keys
                assert 'input_ids' in batch
                assert 'labels' in batch
                assert 'attention_mask' in batch
                
                # Verify shapes (L-1 because of shift)
                assert len(batch['input_ids']) == L - 1
                assert len(batch['labels']) == L - 1
                assert len(batch['attention_mask']) == L - 1
                
                # Verify shift (labels = input_ids shifted by 1)
                expected_input = sequences[idx][:-1]
                expected_labels = sequences[idx][1:]
                expected_mask = masks[idx][:-1]
                
                np.testing.assert_array_equal(batch['input_ids'], expected_input)
                np.testing.assert_array_equal(batch['labels'], expected_labels)
                np.testing.assert_array_equal(batch['attention_mask'], expected_mask)
            
            # Verify all original tokens are preserved (accounting for shift)
            total_input_tokens = sum(len(d) for d in docs)
            total_dataset_tokens = sum((dataset[i]['attention_mask'] == 1).sum() for i in range(len(dataset)))
            
            # Account for the shift: we lose 1 token per sequence due to [:-1]
            # But only if that last token was real (not padding)
            tokens_lost_to_shift = sum(1 for i in range(len(sequences)) if masks[i][-1] == 1)
            expected_tokens = total_input_tokens - tokens_lost_to_shift
            
            assert total_dataset_tokens == expected_tokens, \
                f"Token count mismatch: got {total_dataset_tokens}, expected {expected_tokens} " \
                f"(input={total_input_tokens}, lost_to_shift={tokens_lost_to_shift})"
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_multiple_files(self):
        # Test with multiple files
        temp_dir = tempfile.mkdtemp()
        try:
            data_dir = Path(temp_dir) / "data"
            lang_dir = data_dir / "test_lang"
            lang_dir.mkdir(parents=True)
            
            L = 64
            PAD = 0
            
            # Create and save multiple files
            for file_idx in range(3):
                docs = [[i] * 10 for i in range(file_idx * 10, (file_idx + 1) * 10)]
                sequences, masks = pack_documents(docs, L, PAD)
                
                np.save(lang_dir / f"tokens_{file_idx:05d}.npy", sequences.astype(np.uint32))
                np.save(lang_dir / f"masks_{file_idx:05d}.npy", masks.astype(np.uint8))
            
            # Load dataset
            dataset = MultiLangTokenDataset(data_dir, seq_length=L)
            
            # Should load all files
            assert len(dataset) > 0, "Should load sequences from all files"
            
            # Test random access
            batch = dataset[0]
            assert batch['input_ids'].shape[0] == L - 1
            
            batch = dataset[-1]
            assert batch['labels'].shape[0] == L - 1
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_multiple_languages(self):
        # Test with multiple language directories
        temp_dir = tempfile.mkdtemp()
        try:
            data_dir = Path(temp_dir) / "data"
            
            L = 32
            PAD = 0
            
            # Create multiple language directories
            for lang in ["en", "de", "fr"]:
                lang_dir = data_dir / lang
                lang_dir.mkdir(parents=True)
                
                docs = [[ord(c)] * 5 for c in lang]  # 15 tokens per language
                sequences, masks = pack_documents(docs, L, PAD)
                
                np.save(lang_dir / "tokens_00000.npy", sequences.astype(np.uint32))
                np.save(lang_dir / "masks_00000.npy", masks.astype(np.uint8))
            
            # Load dataset
            dataset = MultiLangTokenDataset(data_dir, seq_length=L)
            
            # Should load from all languages
            assert len(dataset) > 0, "Should load sequences from all languages"
            
        finally:
            shutil.rmtree(temp_dir)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_token_documents(self):
        docs = [[i] for i in range(20)]  # 20 single-token docs
        L = 10
        
        sequences, masks = pack_documents(docs, L, 0)
        
        # Should pack efficiently (2 sequences with 10 tokens each)
        assert len(sequences) == 2
        assert (masks == 1).sum() == 20
    
    def test_exact_length_documents(self):
        docs = [list(range(10)), list(range(10, 20))]  # Exactly L tokens each
        L = 10
        
        sequences, masks = pack_documents(docs, L, 0)
        
        # Each should be its own sequence
        assert len(sequences) == 2
        assert (masks == 1).sum() == 20
        assert np.all(masks == 1), "No padding needed for exact-length docs"
    
    def test_minimal_dataset(self):
        temp_dir = tempfile.mkdtemp()
        try:
            data_dir = Path(temp_dir) / "data"
            lang_dir = data_dir / "test"
            lang_dir.mkdir(parents=True)
            
            L = 16
            
            # Single sequence
            sequences = np.array([[1, 2, 3] + [0] * 13], dtype=np.uint32)
            masks = np.array([[1, 1, 1] + [0] * 13], dtype=np.uint8)
            
            np.save(lang_dir / "tokens_00000.npy", sequences)
            np.save(lang_dir / "masks_00000.npy", masks)
            
            dataset = MultiLangTokenDataset(data_dir, seq_length=L)
            
            assert len(dataset) == 1
            batch = dataset[0]
            assert len(batch['input_ids']) == L - 1
            
        finally:
            shutil.rmtree(temp_dir)


class TestTokenizationPipeline:
    """Test complete tokenization from jsonl.gz to packed sequences"""
    
    def test_tokenize_from_jsonlgz(self):
        temp_dir = tempfile.mkdtemp()
        try:
            # Create test data
            input_dir = Path(temp_dir) / "input" / "test_lang"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir(parents=True)
            
            # Create jsonl.gz file with sample documents
            test_docs = [
                {"text": "Hello world this is a test"},
                {"text": "Another document here"},
                {"text": "Short"},
                {"text": "This is a much longer document that contains many more words"},
            ]
            
            with gzip.open(input_dir / "data.jsonl.gz", "wt", encoding="utf-8") as f:
                for doc in test_docs:
                    f.write(json.dumps(doc) + "\n")
            
            # Create simple tokenizer
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Train on sample text
            all_text = " ".join(doc["text"] for doc in test_docs)
            trainer = trainers.BpeTrainer(vocab_size=100, special_tokens=["<|pad|>", "<|endoftext|>"])
            tokenizer.train_from_iterator([all_text], trainer=trainer)
            
            tokenizer_path = Path(temp_dir) / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            
            # Run tokenization with packing
            L = 32
            tokenize_and_save(
                str(input_dir.parent),
                str(output_dir),
                str(tokenizer_path),
                seq_length=L,
                batch_size=10,
                pack_batch_size=10
            )
            
            # Verify output files exist
            output_lang_dir = output_dir / "test_lang"
            assert output_lang_dir.exists()
            assert list(output_lang_dir.glob("tokens_*.npy"))
            assert list(output_lang_dir.glob("masks_*.npy"))
            
            # Load and verify with dataset
            dataset = MultiLangTokenDataset(output_dir, seq_length=L)
            assert len(dataset) > 0
            
            # Verify batch format
            batch = dataset[0]
            assert batch['input_ids'].shape[0] == L - 1
            assert batch['labels'].shape[0] == L - 1
            assert batch['attention_mask'].shape[0] == L - 1
            
            # Verify some tokens are real (not all padding)
            assert (batch['attention_mask'] == 1).sum() > 0
            
            # Verify data integrity
            for i in range(len(dataset)):
                batch = dataset[i]
                # Where mask is 0, we should eventually see padding
                # (though not necessarily in input_ids due to shift)
                assert batch['input_ids'].dtype == np.int64
                assert batch['labels'].dtype == np.int64
                assert batch['attention_mask'].dtype == np.int64
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

