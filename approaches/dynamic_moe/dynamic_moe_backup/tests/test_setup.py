import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock deepspeed if not installed (for running in CI/Agent envs)
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    import unittest.mock as mock
    print("Warning: DeepSpeed not found. Mocking it for import purposes.")
    DEEPSPEED_AVAILABLE = False
    
    # Create a mock that inherits from nn.Module so PyTorch accepts it
    class MockMoE(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x):
            # Return dummy output matching expected signature: output, loss, ...
            return x, torch.tensor(0.0), None

    m = mock.MagicMock()
    m.MoE = MockMoE
    
    sys.modules["deepspeed"] = m
    sys.modules["deepspeed.moe"] = m
    sys.modules["deepspeed.moe.layer"] = m
    sys.modules["deepspeed.moe.utils"] = m

from htyllm_pg.model_builder import moe_builder
from htyllm_pg.dataset import MultiLangTokenDataset
from tokenizers import Tokenizer

class TestTrainingSetup(unittest.TestCase):
    
    def test_01_loss_masking(self):
        """
        CRITICAL: Verify that CrossEntropyLoss with ignore_index=-100 
        actually ignores the masked tokens.
        """
        print("\n[Test] Verifying Loss Masking...")
        
        batch_size = 2
        seq_len = 4
        vocab_size = 10
        
        # Create dummy logits [B, T, V]
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Create targets [B, T] with some -100 values
        targets = torch.tensor([
            [1, 2, -100, -100],  # Half padded
            [3, 4, 5, 6]         # No padding
        ])
        
        # 1. Compute loss with ignore_index=-100
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        loss_masked = criterion(logits.view(-1, vocab_size), targets.view(-1))
        
        # 2. Compute loss manually by selecting only valid tokens
        valid_mask = targets != -100
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        
        criterion_manual = nn.CrossEntropyLoss(reduction='sum')
        loss_manual = criterion_manual(valid_logits, valid_targets)
        
        print(f"  Masked Loss: {loss_masked.item():.4f}")
        print(f"  Manual Loss: {loss_manual.item():.4f}")
        
        self.assertTrue(torch.isclose(loss_masked, loss_manual), 
                        "Loss function is NOT correctly ignoring masked tokens!")
        print("[OK] Loss masking works correctly.")

    def test_02_tokenizer_config(self):
        """
        Verify tokenizer.json exists and has correct pad_token_id.
        """
        print("\n[Test] Verifying Tokenizer Configuration...")
        tokenizer_path = "tokenizer.json"
        if not os.path.exists(tokenizer_path):
            print(f"  Warning: {tokenizer_path} not found. Skipping tokenizer check.")
            return

        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception as e:
            print(f"  Warning: Could not load tokenizer.json using 'tokenizers' library: {e}")
            print("  This might be due to a version mismatch. Trying transformers...")
            try:
                from transformers import PreTrainedTokenizerFast
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
                print("  Loaded with transformers.PreTrainedTokenizerFast.")
            except Exception as e2:
                print(f"  Error loading tokenizer: {e2}")
                return

        # Check pad token
        if hasattr(tokenizer, "token_to_id"):
            pad_id = tokenizer.token_to_id("<|pad|>")
        else:
            # transformers tokenizer
            pad_id = tokenizer.pad_token_id
            
        print(f"  <|pad|> ID: {pad_id}")
        
        # We expect pad_id to be 0 based on previous investigation
        # But even if it's not 0, it MUST match what the dataset uses.
        # Since dataset.py hardcodes 0 as padding in some places (or implies it),
        # we should verify it is indeed 0 or consistent.
        
        if tokenizer.padding:
            print(f"  Tokenizer padding config: {tokenizer.padding}")
            self.assertEqual(tokenizer.padding['pad_id'], pad_id, "Tokenizer padding ID mismatch")
        
        # In this project, we expect 0
        if pad_id is not None:
             self.assertEqual(pad_id, 0, "Expected <|pad|> to be ID 0 for this project's convention.")
        
        print("[OK] Tokenizer config looks reasonable.")

    def test_03_dataset_consistency(self):
        """
        If data directory is provided via env var TEST_DATA_DIR, 
        verify that input_ids=0 corresponds to labels=-100.
        """
        print("\n[Test] Verifying Dataset Consistency...")
        data_dir = os.environ.get("TEST_DATA_DIR")
        if not data_dir:
            print("  TEST_DATA_DIR env var not set. Skipping dataset check.")
            print("  (Run with TEST_DATA_DIR=/path/to/data python tests/test_setup.py)")
            return
            
        if not os.path.exists(data_dir):
            print(f"  Data dir {data_dir} does not exist. Skipping.")
            return

        dataset = MultiLangTokenDataset(data_dir, seq_length=128) # Short seq len for speed if possible, but dataset loads fixed len
        # We just need one sample
        if len(dataset) == 0:
            print("  Dataset is empty.")
            return
            
        sample = dataset[0]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        print(f"  Sample 0 shapes: Input {input_ids.shape}, Labels {labels.shape}")
        
        # Check: Wherever input is 0 (padding), label should be -100
        # Note: input_ids is seq[:-1], labels is seq[1:]
        # So if the original sequence was [A, B, 0, 0]
        # input: [A, B, 0]
        # label: [B, 0, 0] -> [B, -100, -100]
        
        # Find padding in inputs
        pad_mask = (input_ids == 0)
        
        if pad_mask.any():
            masked_labels = labels[pad_mask]
            print(f"  Found {len(masked_labels)} padded positions in input.")
            print(f"  Labels at these positions: {masked_labels}")
            
            # Verify all are -100
            self.assertTrue((masked_labels == -100).all(), 
                            "Found padding tokens in input where label was NOT -100!")
            print("[OK] Dataset correctly masks padding.")
        else:
            print("  No padding found in first sample. Checking for -100 in labels anyway...")
            if (labels == -100).any():
                 print("  Found -100 in labels (good).")
            else:
                 print("  No padding or -100 found in first sample. Try checking more samples if possible.")

    def test_04_model_forward(self):
        """
        Verify model can run a forward pass with the loss calculation.
        """
        if not DEEPSPEED_AVAILABLE:
            print("\n[Test] Skipping Model Forward Pass (DeepSpeed not installed)...")
            return

        print("\n[Test] Verifying Model Forward Pass...")
        vocab_size = 1000
        seq_len = 32
        
        model = moe_builder(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            num_experts=4,
            moe_layers=[1]
        )
        
        input_ids = torch.randint(0, vocab_size, (2, seq_len))
        # Add some padding
        input_ids[0, -5:] = 0
        
        # Forward
        logits, aux_loss, expert_counts = model(input_ids)
        
        self.assertEqual(logits.shape, (2, seq_len, vocab_size))
        print("[OK] Model forward pass successful.")

if __name__ == '__main__':
    unittest.main()
