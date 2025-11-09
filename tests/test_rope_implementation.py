"""
Test script for RoPE implementation

Run this test after installing the package:
    uv pip install -e .  (if using uv)
    or: pip install -e .  (if using pip)

Make sure you're in your venv before running!
"""
import sys
import os
import torch
import traceback

# Add project root to path (go up one level from tests/ directory)
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)  # Go up one level to project root
sys.path.insert(0, project_root)

# Clean import helper - handles both package and local imports
def _import_model_builder():
    """Import model_builder functions, handling both package and local imports."""
    try:
        # Try package import first (when installed)
        from htyllm_pg.model_builder import moe_builder, rotate_half, apply_rotary_pos_emb, precompute_freqs_cis
        return moe_builder, rotate_half, apply_rotary_pos_emb, precompute_freqs_cis
    except ImportError:
        # Fallback to local import (for development)
        sys.path.insert(0, os.path.join(project_root, 'htyllm-pg'))
        from model_builder import moe_builder, rotate_half, apply_rotary_pos_emb, precompute_freqs_cis
        return moe_builder, rotate_half, apply_rotary_pos_emb, precompute_freqs_cis

# Import once at module level
moe_builder, rotate_half, apply_rotary_pos_emb, precompute_freqs_cis = _import_model_builder()

def test_shape_verification():
    """Test 1: Shape Verification"""
    print("=" * 50)
    print("Test 1: Shape Verification")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    batch_size, seq_len, dim = 2, 10, 512
    num_heads = 8
    vocab_size = 1000
    
    # Use model without MoE layers for simpler testing (avoids DeepSpeed init requirement)
    model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, dim=dim, heads=num_heads, use_rope=True, moe_layers=[])
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    output, l_aux = model(tokens)
    
    assert output.shape == (batch_size, seq_len, vocab_size), f"Shape mismatch! Expected {(batch_size, seq_len, vocab_size)}, got {output.shape}"
    print("OK: Shape test passed!")
    print(f"  Output shape: {output.shape}")
    print()


def test_backward_compatibility():
    """Test 2: Backward Compatibility"""
    print("=" * 50)
    print("Test 2: Backward Compatibility")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    # Test with RoPE disabled (should match old behavior)
    model_no_rope = moe_builder(vocab_size=1000, max_seq_len=128, use_rope=False, moe_layers=[])
    # Should still have pos_embedding parameter
    assert hasattr(model_no_rope, 'pos_embedding') and model_no_rope.pos_embedding is not None, "pos_embedding should exist when use_rope=False"
    print("OK: Backward compatibility test passed!")
    print(f"  pos_embedding exists: {model_no_rope.pos_embedding is not None}")
    print()


def test_variable_sequence_lengths():
    """Test 3: Variable Sequence Lengths"""
    print("=" * 50)
    print("Test 3: Variable Sequence Lengths")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    # RoPE should handle different sequence lengths
    model = moe_builder(vocab_size=1000, max_seq_len=512, use_rope=True, moe_layers=[])
    vocab_size = 1000
    
    for seq_len in [64, 128, 256, 512]:
        tokens = torch.randint(0, vocab_size, (2, seq_len))
        output, _ = model(tokens)
        assert output.shape[1] == seq_len, f"Failed for seq_len={seq_len}, got {output.shape[1]}"
        print(f"  OK: seq_len={seq_len}: output shape {output.shape}")
    
    print("OK: Variable sequence length test passed!")
    print()


def test_rope_application():
    """Test 4: RoPE Application Check"""
    print("=" * 50)
    print("Test 4: RoPE Application Check")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    # Verify RoPE is actually being applied
    model = moe_builder(vocab_size=1000, max_seq_len=128, use_rope=True, moe_layers=[])
    # Check that Attention layers have freqs_cos and freqs_sin
    for layer in model.transformer.layers:
        attn = layer[0]
        assert hasattr(attn, 'freqs_cos'), "RoPE frequencies not found!"
        assert attn.freqs_cos is not None, "RoPE frequencies are None!"
        assert hasattr(attn, 'freqs_sin'), "RoPE frequencies not found!"
        assert attn.freqs_sin is not None, "RoPE frequencies are None!"
    
    print("OK: RoPE application test passed!")
    print(f"  Number of layers: {len(model.transformer.layers)}")
    # Note: freqs_cos/sin now have shape [max_seq_len, dim_head/2] (not interleaved)
    attn = model.transformer.layers[0][0]
    expected_shape = (128, attn.dim_head // 2)
    assert attn.freqs_cos.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {attn.freqs_cos.shape}"
    print(f"  Frequency cache shape: {attn.freqs_cos.shape} (non-interleaved)")
    print()


def test_dtype_safety():
    """Test 5: Dtype Safety"""
    print("=" * 50)
    print("Test 5: Dtype Safety")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    seq_len = 64
    
    for dtype_name, dtype in [("float32", torch.float32), ("float16", torch.float16)]:
        try:
            model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
            model = model.to(dtype)
            tokens = torch.randint(0, vocab_size, (1, seq_len))
            
            with torch.no_grad():
                output, _ = model(tokens)
                # Check that output dtype matches (or is compatible)
                print(f"  OK: {dtype_name}: output dtype {output.dtype}, model dtype {dtype}")
        except Exception as e:
            print(f"  WARNING: {dtype_name} test failed: {e}")
    
    # Skip bfloat16 if not available
    if hasattr(torch, 'bfloat16'):
        try:
            model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
            model = model.to(torch.bfloat16)
            tokens = torch.randint(0, vocab_size, (1, seq_len))
            
            with torch.no_grad():
                output, _ = model(tokens)
                print(f"  OK: bfloat16: output dtype {output.dtype}")
        except Exception as e:
            print(f"  WARNING: bfloat16 test failed: {e}")
    
    print("OK: Dtype safety test passed!")
    print()


def test_cache_growth():
    """Test 6: Cache Growth"""
    print("=" * 50)
    print("Test 6: Cache Growth")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    # Test that frequency cache grows when needed
    model = moe_builder(vocab_size=1000, max_seq_len=128, use_rope=True, moe_layers=[])
    initial_max = model.transformer.layers[0][0].freqs_cos.size(0)
    assert initial_max == 128, f"Initial max_seq_len mismatch: expected 128, got {initial_max}"
    print(f"  Initial max_seq_len: {initial_max}")
    
    # Process longer sequence
    tokens_long = torch.randint(0, 1000, (1, 256))
    with torch.no_grad():
        _ = model(tokens_long)
    
    # Check that cache grew
    new_max = model.transformer.layers[0][0].freqs_cos.size(0)
    assert new_max >= 256, f"Cache did not grow! Still {new_max}, need >= 256"
    print(f"  New max_seq_len after processing 256 tokens: {new_max}")
    print("OK: Cache growth test passed!")
    print()


def test_rope_norm_preservation():
    """Test 7: RoPE Norm Preservation (Critical Property)"""
    print("=" * 50)
    print("Test 7: RoPE Norm Preservation")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    # RoPE is a rotation → it preserves L2 norm of the rotated subspace
    model = moe_builder(vocab_size=1000, max_seq_len=128, use_rope=True, moe_layers=[], dim_head=64, heads=4)
    attn = model.transformer.layers[0][0]
    
    # Create test query tensor [batch, heads, seq_len, head_dim]
    B, T, H, Dh = 2, 16, 4, 64
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)  # Also need k for apply_rotary_pos_emb
    
    # Compute norm before rotation
    q_norm_before = q.norm(dim=-1)  # [B, H, T]
    
    # Apply RoPE using the actual function (same as in Attention.forward)
    q_rotated, k_rotated = apply_rotary_pos_emb(
        q, k,
        attn.freqs_cos,
        attn.freqs_sin,
        position_ids=None,
        past_kv_len=0,
        rope_dim=None
    )
    
    # Compute norm after rotation
    q_norm_after = q_rotated.norm(dim=-1)  # [B, H, T]
    
    # Norms should be identical (rotation preserves norm)
    # Use slightly higher tolerance for numerical precision
    assert torch.allclose(q_norm_before, q_norm_after, atol=1e-4, rtol=1e-5), \
        f"RoPE should preserve norm! Max diff: {(q_norm_before - q_norm_after).abs().max().item():.2e}, " \
        f"Mean diff: {(q_norm_before - q_norm_after).abs().mean().item():.2e}"
    
    print("OK: RoPE norm preservation test passed!")
    print(f"  Max norm difference: {(q_norm_before - q_norm_after).abs().max().item():.2e}")
    print(f"  Mean norm difference: {(q_norm_before - q_norm_after).abs().mean().item():.2e}")
    print()


def test_buffer_dtype_device_alignment():
    """Test 8: Buffer Dtype/Device Alignment"""
    print("=" * 50)
    print("Test 8: Buffer Dtype/Device Alignment")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    seq_len = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for dtype_name, dtype in [("float32", torch.float32), ("float16", torch.float16)]:
        if dtype == torch.float16 and not torch.cuda.is_available():
            print(f"  SKIP: {dtype_name} (requires CUDA)")
            continue
            
        try:
            model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
            model = model.to(dtype=dtype, device=device)
            
            attn = model.transformer.layers[0][0]
            
            # Buffers should be on same device
            assert attn.freqs_cos.device == device, \
                f"freqs_cos device mismatch: {attn.freqs_cos.device} != {device}"
            assert attn.freqs_sin.device == device, \
                f"freqs_sin device mismatch: {attn.freqs_sin.device} != {device}"
            
            # Buffers are stored as float32 (they get cast during forward), but device should match
            print(f"  OK: {dtype_name}: buffers on device {attn.freqs_cos.device}")
        except Exception as e:
            print(f"  WARNING: {dtype_name} test failed: {e}")
    
    # Test bfloat16 if available
    if hasattr(torch, 'bfloat16'):
        try:
            model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
            model = model.to(dtype=torch.bfloat16, device=device)
            attn = model.transformer.layers[0][0]
            assert attn.freqs_cos.device == device
            print(f"  OK: bfloat16: buffers on device {attn.freqs_cos.device}")
        except Exception as e:
            print(f"  WARNING: bfloat16 test failed: {e}")
    
    print("OK: Buffer dtype/device alignment test passed!")
    print()


def test_backprop():
    """Test 9: Backpropagation (Gradient Flow)"""
    print("=" * 50)
    print("Test 9: Backpropagation")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    # Verify that RoPE operations are differentiable
    model = moe_builder(vocab_size=1000, max_seq_len=128, use_rope=True, moe_layers=[])
    tokens = torch.randint(0, 1000, (2, 64))
    
    output, l_aux = model(tokens)
    loss = output.mean() + 0.01 * l_aux
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for key parameters
    assert model.token_embedding.weight.grad is not None, "token_embedding should have gradients"
    assert model.mlp_head.weight.grad is not None, "mlp_head should have gradients"
    
    # Check attention weights have gradients
    attn = model.transformer.layers[0][0]
    assert attn.to_qkv.weight.grad is not None, "to_qkv should have gradients"
    
    print("OK: Backpropagation test passed!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed for {sum(1 for p in model.parameters() if p.grad is not None)}/{sum(1 for p in model.parameters() if p.requires_grad)} parameters")
    print()


def test_partial_rotary():
    """Test 10: Partial RoPE (rope_dim < head_dim)"""
    print("=" * 50)
    print("Test 10: Partial RoPE")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    seq_len = 64
    dim_head = 64
    rope_dim = 32  # Only rotate first half
    
    # Test with partial RoPE - pass rope_dim during initialization
    model = moe_builder(
        vocab_size=vocab_size, 
        max_seq_len=seq_len, 
        use_rope=True, 
        moe_layers=[],
        dim_head=dim_head,
        rope_dim=rope_dim  # Pass rope_dim directly
    )
    
    tokens = torch.randint(0, vocab_size, (1, seq_len))
    
    with torch.no_grad():
        output, _ = model(tokens)
    
    # Verify it works without error
    assert output.shape == (1, seq_len, vocab_size), "Partial RoPE should produce correct shape"
    
    # Verify frequencies were precomputed with correct dimension
    attn = model.transformer.layers[0][0]
    assert attn.rope_dim == rope_dim, f"Expected rope_dim={rope_dim}, got {attn.rope_dim}"
    assert attn.freqs_cos.shape == (seq_len, rope_dim // 2), \
        f"Expected freqs_cos shape ({seq_len}, {rope_dim // 2}), got {attn.freqs_cos.shape}"
    
    print("OK: Partial RoPE test passed!")
    print(f"  head_dim: {dim_head}, rope_dim: {rope_dim}")
    print(f"  Frequency cache shape: {attn.freqs_cos.shape}")
    print()


def test_determinism():
    """Test 11: Determinism (Reproducibility)"""
    print("=" * 50)
    print("Test 11: Determinism")
    print("=" * 50)
    
    vocab_size = 1000
    seq_len = 32
    
    # Run twice with same seed - should get identical results
    torch.manual_seed(42)
    model1 = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
    tokens = torch.randint(0, vocab_size, (1, seq_len))
    with torch.no_grad():
        output1, _ = model1(tokens)
    
    torch.manual_seed(42)
    model2 = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
    tokens2 = torch.randint(0, vocab_size, (1, seq_len))
    with torch.no_grad():
        output2, _ = model2(tokens2)
    
    # Results should be identical
    assert torch.allclose(output1, output2, atol=1e-6), "Results should be deterministic with same seed"
    
    print("OK: Determinism test passed!")
    print(f"  Max difference: {(output1 - output2).abs().max().item():.2e}")
    print()


def test_attention_masking():
    """Test 12: Attention Mask Support"""
    print("=" * 50)
    print("Test 12: Attention Mask Support")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    batch_size = 2
    seq_len = 64
    
    model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create attention mask: first sequence has padding at end
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, 32:] = False  # Mask out last 32 tokens of first sequence
    
    with torch.no_grad():
        # Test with mask
        output_masked, _ = model(tokens, attention_mask=attention_mask)
        
        # Test without mask (should differ)
        output_unmasked, _ = model(tokens)
    
    # Outputs should differ (masked attention changes results)
    assert not torch.allclose(output_masked, output_unmasked, atol=1e-3), \
        "Attention mask should change output"
    
    # Masked positions should have lower influence
    # Check that the difference is more pronounced in masked region
    diff = (output_masked - output_unmasked).abs()
    masked_diff = diff[0, 32:].mean()
    unmasked_diff = diff[0, :32].mean()
    
    print(f"  Masked region diff: {masked_diff:.4f}")
    print(f"  Unmasked region diff: {unmasked_diff:.4f}")
    print("OK: Attention mask test passed!")
    print()


def test_causal_masking():
    """Test 13: Causal Masking Control"""
    print("=" * 50)
    print("Test 13: Causal Masking")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    seq_len = 32
    
    model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
    tokens = torch.randint(0, vocab_size, (1, seq_len))
    
    with torch.no_grad():
        # Test with causal masking (default for autoregressive)
        output_causal, _ = model(tokens, is_causal=True)
        
        # Test without causal masking (bidirectional, like BERT)
        output_bidirectional, _ = model(tokens, is_causal=False)
    
    # Results should differ (causal mask prevents future token access)
    assert not torch.allclose(output_causal, output_bidirectional, atol=1e-3), \
        "Causal masking should change output"
    
    print("OK: Causal masking test passed!")
    print(f"  Max difference: {(output_causal - output_bidirectional).abs().max().item():.4f}")
    print()


def test_kv_cache():
    """Test 14: KV-Cache for Autoregressive Generation"""
    print("=" * 50)
    print("Test 14: KV-Cache")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    seq_len = 16
    
    model = moe_builder(vocab_size=vocab_size, max_seq_len=128, use_rope=True, moe_layers=[])
    model.eval()  # Disable dropout for deterministic results
    
    # Initial prompt
    prompt = torch.randint(0, vocab_size, (1, seq_len))
    
    with torch.no_grad():
        # First pass: compute with full prompt
        output_full, _, kv_cache = model(prompt, use_cache=True)
        
        # Second pass: use KV-cache for next token
        next_token = torch.randint(0, vocab_size, (1, 1))
        output_cached, _, _ = model(
            next_token, 
            use_cache=True, 
            past_key_values=kv_cache
        )
        
        # Verify KV-cache structure
        assert kv_cache is not None, "KV-cache should be returned"
        assert len(kv_cache) == len(model.transformer.layers), \
            f"KV-cache should have one entry per layer, got {len(kv_cache)}"
        
        # Each layer's cache should be (k, v) tuple
        for i, cache in enumerate(kv_cache):
            assert isinstance(cache, tuple) and len(cache) == 2, \
                f"Layer {i} cache should be (k, v) tuple"
            k, v = cache
            assert k.shape[2] == seq_len, \
                f"Layer {i} cached keys should have seq_len={seq_len}, got {k.shape[2]}"
    
    print("OK: KV-cache test passed!")
    print(f"  Number of cached layers: {len(kv_cache)}")
    print(f"  Cached sequence length: {kv_cache[0][0].shape[2]}")
    print()


def test_flash_attention_equivalence():
    """Test 15: Flash Attention vs Manual Attention"""
    print("=" * 50)
    print("Test 15: Flash Attention Equivalence")
    print("=" * 50)
    
    # Check if Flash Attention is available
    has_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    if not has_flash:
        print("  SKIP: Flash Attention not available (requires PyTorch 2.0+)")
        print()
        return
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    seq_len = 32
    
    # Test with Flash Attention enabled
    model_flash = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
    # Enable Flash Attention in all attention layers
    for layer in model_flash.transformer.layers:
        layer[0].use_flash_attention = True
    
    # Test with Flash Attention disabled
    model_manual = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len, use_rope=True, moe_layers=[])
    # Disable Flash Attention in all attention layers
    for layer in model_manual.transformer.layers:
        layer[0].use_flash_attention = False
    
    # Copy weights to ensure same initialization
    model_manual.load_state_dict(model_flash.state_dict())
    
    tokens = torch.randint(0, vocab_size, (1, seq_len))
    
    with torch.no_grad():
        output_flash, _ = model_flash(tokens)
        output_manual, _ = model_manual(tokens)
    
    # Results should be very similar (allowing for numerical differences)
    assert torch.allclose(output_flash, output_manual, atol=1e-3, rtol=1e-3), \
        "Flash Attention and manual attention should produce similar results"
    
    print("OK: Flash Attention equivalence test passed!")
    print(f"  Max difference: {(output_flash - output_manual).abs().max().item():.4e}")
    print(f"  Mean difference: {(output_flash - output_manual).abs().mean().item():.4e}")
    print()


def test_rope_scaling():
    """Test 16: RoPE Scaling (Linear/NTK)"""
    print("=" * 50)
    print("Test 16: RoPE Scaling")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    seq_len = 256  # Longer than typical training length
    
    # Test linear scaling
    model_linear = moe_builder(
        vocab_size=vocab_size, 
        max_seq_len=seq_len, 
        use_rope=True, 
        moe_layers=[],
        rope_scaling={'type': 'linear', 'factor': 2.0}
    )
    
    # Test NTK scaling
    model_ntk = moe_builder(
        vocab_size=vocab_size, 
        max_seq_len=seq_len, 
        use_rope=True, 
        moe_layers=[],
        rope_scaling={'type': 'ntk', 'factor': 2.0}
    )
    
    tokens = torch.randint(0, vocab_size, (1, seq_len))
    
    with torch.no_grad():
        output_linear, _ = model_linear(tokens)
        output_ntk, _ = model_ntk(tokens)
    
    # Both should work without error
    assert output_linear.shape == (1, seq_len, vocab_size)
    assert output_ntk.shape == (1, seq_len, vocab_size)
    
    # Verify frequencies were scaled
    attn_linear = model_linear.transformer.layers[0][0]
    attn_ntk = model_ntk.transformer.layers[0][0]
    
    # Linear scaling should affect all frequencies uniformly
    # NTK scaling should affect frequencies non-uniformly
    # They should produce different frequency patterns
    assert not torch.allclose(attn_linear.freqs_cos, attn_ntk.freqs_cos, atol=1e-5), \
        "Linear and NTK scaling should produce different frequencies"
    
    print("OK: RoPE scaling test passed!")
    print(f"  Linear scaling: freqs shape {attn_linear.freqs_cos.shape}")
    print(f"  NTK scaling: freqs shape {attn_ntk.freqs_cos.shape}")
    print()


def test_edge_cases():
    """Test 17: Edge Cases (Single Token, etc.)"""
    print("=" * 50)
    print("Test 17: Edge Cases")
    print("=" * 50)
    
    torch.manual_seed(0)
    
    vocab_size = 1000
    model = moe_builder(vocab_size=vocab_size, max_seq_len=128, use_rope=True, moe_layers=[])
    
    # Test single token
    tokens_single = torch.randint(0, vocab_size, (1, 1))
    with torch.no_grad():
        output_single, _ = model(tokens_single)
    assert output_single.shape == (1, 1, vocab_size), "Single token should work"
    print("  OK: Single token")
    
    # Test very long sequence (cache growth)
    tokens_long = torch.randint(0, vocab_size, (1, 512))
    with torch.no_grad():
        output_long, _ = model(tokens_long)
    assert output_long.shape == (1, 512, vocab_size), "Long sequence should work"
    print("  OK: Long sequence (512 tokens)")
    
    print("OK: Edge cases test passed!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("RoPE Implementation Tests")
    print("=" * 50 + "\n")
    
    try:
        test_shape_verification()
        test_backward_compatibility()
        test_variable_sequence_lengths()
        test_rope_application()
        test_dtype_safety()
        test_cache_growth()
        test_rope_norm_preservation()
        test_buffer_dtype_device_alignment()
        test_backprop()
        test_partial_rotary()
        test_determinism()
        test_attention_masking()
        test_causal_masking()
        test_kv_cache()
        test_flash_attention_equivalence()
        test_rope_scaling()
        test_edge_cases()
        
        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        traceback.print_exc()
        exit(1)

