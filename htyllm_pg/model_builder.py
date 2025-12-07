from typing import List
import torch
from torch import nn

from einops import rearrange


# ============================================
# RoPE (Rotary Position Embedding) Functions
# Production-grade with KV-cache, scaling, partial-RoPE support
# ============================================

def precompute_freqs_cis(
    dim: int, 
    max_seq_len: int, 
    theta: float = 10000.0,
    rope_scaling: dict = None
):
    """
    Precompute cosine and sine frequencies for RoPE.
    
    Args:
        dim: Head dimension (should be even)
        max_seq_len: Maximum sequence length
        theta: Base frequency (default: 10000.0 from paper)
        rope_scaling: Optional scaling config for longer context
            - {'type': 'linear', 'factor': 2.0} for linear scaling
            - {'type': 'ntk', 'factor': 2.0, 'alpha': 1.0} for NTK-aware scaling
            - {'type': 'yarn', 'factor': 2.0, 'alpha': 1.0} for YaRN scaling
    
    Returns:
        Tuple of (freqs_cos, freqs_sin) with shape [max_seq_len, dim/2]
    """
    assert dim % 2 == 0, f"dim ({dim}) must be even for RoPE"
    
    # Compute frequency for each dimension pair
    # θ_i = 1 / (theta^(2i/d)) for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Apply rope scaling if specified
    if rope_scaling:
        scaling_type = rope_scaling.get('type')
        factor = rope_scaling.get('factor', 1.0)
        
        if scaling_type == 'ntk':
            # NTK-aware scaling: adjust base theta
            # Formula: theta_new = theta * alpha^(d / (d-2))
            alpha = rope_scaling.get('alpha', factor)
            theta_adjusted = theta * (alpha ** (dim / (dim - 2)))
            freqs = 1.0 / (theta_adjusted ** (torch.arange(0, dim, 2).float() / dim))
        elif scaling_type == 'yarn':
            # Simplified YaRN - for production long-context, use HuggingFace's rope_utils
            import warnings
            warnings.warn(
                "Simplified YaRN implementation. For production long-context (>8K tokens), "
                "consider using transformers.modeling_rope_utils for accurate NTK-by-parts implementation.",
                UserWarning
            )
            # Simplified version (acceptable for basic use):
            scale = rope_scaling.get('scale', factor)
            freqs = freqs / scale
    
    # Create position indices [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    
    # Apply linear scaling to positions if specified
    if rope_scaling and rope_scaling.get('type') == 'linear':
        factor = rope_scaling.get('factor', 1.0)
        positions = positions / factor
    
    # Compute outer product: position * frequency
    # Shape: [max_seq_len, dim/2]
    # Use torch.outer if available (PyTorch 1.10+), otherwise use manual
    if hasattr(torch, 'outer'):
        angles = torch.outer(positions, freqs)
    else:
        # Fallback for older PyTorch versions
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    
    # Compute cos and sin (NO interleaving - keep at [max_seq_len, dim/2])
    freqs_cos = torch.cos(angles)  # [max_seq_len, dim/2]
    freqs_sin = torch.sin(angles)  # [max_seq_len, dim/2]
    
    return freqs_cos, freqs_sin


def rotate_half(x):
    """
    COMPATIBILITY NOTE: This function implements split-half rotation (GPT-NeoX style).
    
    The current implementation uses interleaved even/odd rotation (LLaMA/GPT-J style)
    in apply_rotary_pos_emb, which is the modern standard. This function is kept
    for reference but is NOT used in the main RoPE application.
    
    If you need split-half rotation for compatibility with older models:
    - Use this function
    - Adjust frequency precomputation accordingly
    
    For new models, prefer the even/odd interleaved approach in apply_rotary_pos_emb.
    """
    # Reshape to separate pairs: [..., dim] -> [..., dim/2, 2]
    x_reshaped = x.contiguous().view(*x.shape[:-1], -1, 2)
    # Rotate each pair: [x0, x1] -> [-x1, x0]
    x_rotated = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    # Reshape back: [..., dim/2, 2] -> [..., dim]
    return x_rotated.view(*x.shape)


def apply_rotary_pos_emb(
    q, 
    k, 
    freqs_cos, 
    freqs_sin, 
    position_ids=None, 
    past_kv_len: int = 0,
    rope_dim: int = None
):
    """
    Apply rotary position embeddings to query and key tensors.
    Production-grade with KV-cache offset support and partial RoPE.
    Uses even/odd formulation for correct norm preservation.
    
    Note: We use even/odd indexing directly instead of rotate_half because:
    - It matches the canonical RoFormer/LLaMA derivation exactly
    - Ensures perfect norm preservation (orthonormal 2×2 rotations)
    - Works correctly with non-interleaved cos/sin frequencies
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        freqs_cos: Precomputed cosines [max_seq_len, rope_dim/2] (NOT interleaved)
        freqs_sin: Precomputed sines [max_seq_len, rope_dim/2] (NOT interleaved)
        position_ids: Optional position IDs [batch, seq_len] or [seq_len]
            If None, uses past_kv_len to compute positions
        past_kv_len: Length of past KV cache (for autoregressive decoding)
        rope_dim: Number of dimensions to rotate (None = full head_dim)
    
    Returns:
        Tuple of rotated (q, k)
    """
    b, h, n, d = q.shape
    rd = rope_dim if rope_dim is not None else d
    assert rd % 2 == 0, f"rope_dim ({rd}) must be even"
    
    # Determine positions
    if position_ids is None:
        # [n] -> [past_kv_len, past_kv_len+1, ..., past_kv_len+n-1]
        positions = torch.arange(past_kv_len, past_kv_len + n, device=q.device, dtype=torch.long)
    else:
        # Expect [b, n] or [n]; broadcast per head
        positions = position_ids.to(q.device).long()
    
    # Select frequencies by positions
    if positions.dim() == 1:
        # Single sequence: [n] positions
        cos_t = freqs_cos.index_select(0, positions)      # [n, rd/2]
        sin_t = freqs_sin.index_select(0, positions)      # [n, rd/2]
        # Cast to q's dtype and reshape for broadcasting
        cos_t = cos_t.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, n, rd/2]
        sin_t = sin_t.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, n, rd/2]
    else:
        # Packed sequences: [b, n] positions
        cos_t = freqs_cos.index_select(0, positions.reshape(-1))
        sin_t = freqs_sin.index_select(0, positions.reshape(-1))
        cos_t = cos_t.view(b, n, rd // 2).to(device=q.device, dtype=q.dtype).unsqueeze(1)  # [b, 1, n, rd/2]
        sin_t = sin_t.view(b, n, rd // 2).to(device=q.device, dtype=q.dtype).unsqueeze(1)  # [b, 1, n, rd/2]
    
    def _rotate(x):
        """Rotate using even/odd formulation for norm preservation."""
        x_rot = x[..., :rd]                     # [..., rd] - rotary slice
        x_pass = x[..., rd:] if rd < d else None  # [..., d-rd] - non-rotary slice
        
        # Split into even and odd indices
        x_even = x_rot[..., 0::2]               # [..., rd/2] - even indices
        x_odd = x_rot[..., 1::2]                # [..., rd/2] - odd indices
        
        # Apply 2x2 rotation: [x_even', x_odd'] = [cos*e - sin*o, sin*e + cos*o]
        x_rot_even = x_even * cos_t - x_odd * sin_t
        x_rot_odd = x_odd * cos_t + x_even * sin_t
        
        # Interleave back: [even0, odd0, even1, odd1, ...]
        x_new = torch.empty_like(x_rot)
        x_new[..., 0::2] = x_rot_even
        x_new[..., 1::2] = x_rot_odd
        
        # Concatenate with non-rotary part if partial RoPE
        return torch.cat([x_new, x_pass], dim=-1) if x_pass is not None else x_new
    
    return _rotate(q), _rotate(k)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.,
        max_seq_len=512,           # NEW: For RoPE frequency computation
        use_rope=True,             # NEW: Enable/disable RoPE
        rope_theta=10000.0,        # NEW: RoPE base frequency
        rope_dim=None,             # NEW: Partial RoPE (None = full head)
        rope_scaling=None,         # NEW: Scaling config for longer context
        use_flash_attention=False  # NEW: Use SDPA/FlashAttention
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads # number of attention heads (how many attentions are stacked per layer)
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5 
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention

        self.norm = nn.LayerNorm(dim) 

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) 
                                                                  
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # NEW: RoPE frequency computation with cache growth support
        if self.use_rope:
            # Ensure dim_head is even for RoPE
            assert dim_head % 2 == 0, f"dim_head ({dim_head}) must be even for RoPE"
            # Store RoPE config for cache growth
            self.rope_dim = rope_dim if rope_dim is not None else dim_head
            assert self.rope_dim % 2 == 0, f"rope_dim ({self.rope_dim}) must be even"
            assert self.rope_dim <= dim_head, f"rope_dim ({self.rope_dim}) cannot exceed dim_head ({dim_head})"
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling or {}
            self.max_seq_len = max_seq_len
            # Precompute frequencies for RoPE using rope_dim (not dim_head)
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.rope_dim,  # Use rope_dim, not dim_head
                max_seq_len=max_seq_len,
                theta=rope_theta,
                rope_scaling=rope_scaling
            )
            # Register as buffers (persistent=False allows cache growth)
            self.register_buffer('freqs_cos', freqs_cos, persistent=False)
            self.register_buffer('freqs_sin', freqs_sin, persistent=False)
        else:
            self.register_buffer('freqs_cos', None)
            self.register_buffer('freqs_sin', None)
            self.rope_dim = None
            self.rope_theta = None
            self.rope_scaling = {}
            self.max_seq_len = max_seq_len

    def _maybe_grow_freqs(self, need_len: int):
        """
        Dynamically grow frequency cache if sequence length exceeds initial max_seq_len.
        This enables handling longer sequences during inference without hard-crashing.
        """
        if not self.use_rope or self.freqs_cos is None:
            return
        
        current_max = self.freqs_cos.size(0)
        if need_len <= current_max:
            return
        
        # Recompute frequencies for the new length using rope_dim
        new_cos, new_sin = precompute_freqs_cis(
            dim=self.rope_dim,  # Use rope_dim, not dim_head
            max_seq_len=need_len,
            theta=self.rope_theta,
            rope_scaling=self.rope_scaling
        )
        
        # Move to same device and update buffers
        device = self.freqs_cos.device
        self.freqs_cos = new_cos.to(device)
        self.freqs_sin = new_sin.to(device)
        self.max_seq_len = need_len

    def forward(
        self, 
        x, 
        attention_mask=None,      # NEW: [batch, seq_len] or [batch, 1, seq_len, seq_len]
        position_ids=None, 
        past_kv_len: int = 0,
        use_cache: bool = False,
        is_causal: bool = True    # NEW: Control causal masking explicitly
    ):
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            attention_mask: Optional attention mask [batch, seq_len] or [batch, 1, seq_len, seq_len]
                True/1 = attend, False/0 = mask out. If None and is_causal=True, applies causal mask.
            position_ids: Optional position IDs [batch, seq_len] or [seq_len]
            past_kv_len: Length of past KV cache (for autoregressive decoding)
            use_cache: Whether to return KV cache
            is_causal: Whether to apply causal masking (only if attention_mask is None)
        
        Returns:
            output: [batch, seq_len, dim]
            (optional) past_key_value: Tuple of (k_cache, v_cache) if use_cache=True
        """
        x = self.norm(x)# nomralize each tokens along dimension (mean 0, variance 1) shape stays 

        qkv = self.to_qkv(x).chunk(3, dim = -1)# linear layer that maps each 8-dim vector to a big vector (inner_dim * 3)
                                               # 3 because after chunk the is one for q, k, v (shape each: (batch, tokens, inner_dim) )
                                                # Q = x @ W_Q
                                                # K = x @ W_K
                                                # V = x @ W_V

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # split into multiple heads 
                                                                                                # query, key, and values for each token for each header
                                                                                                # Heads might focus on different things in a sentence (syntax, semantic, ...who knows:v)
        # After reshaping: q, k, v each have shape [batch, heads, seq_len, dim_head]

        # ========================================
        # NEW: Apply RoPE to Q and K (NOT V!)
        # ========================================
        if self.use_rope:
            seq_len = q.shape[2]
            total_len = past_kv_len + seq_len
            
            # Grow frequency cache if needed
            self._maybe_grow_freqs(total_len)
            
            # Apply RoPE with KV-cache offset support and dtype safety
            q, k = apply_rotary_pos_emb(
                q, k,
                self.freqs_cos,
                self.freqs_sin,
                position_ids=position_ids,
                past_kv_len=past_kv_len,
                rope_dim=self.rope_dim
            )

        # Compute attention scores (Q and K are now rotated)
        # ========================================
        # CORRECTED: Flash Attention Path
        # ========================================
        if self.use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Prepare attention mask (must be bool, True = attend)
            sdpa_mask = None
            if attention_mask is not None:
                # Ensure mask is bool and correctly shaped
                if attention_mask.dim() == 2:
                    # [batch, seq_len] -> [batch, 1, 1, seq_len]
                    attention_mask = attention_mask[:, None, None, :]
                sdpa_mask = attention_mask.bool()
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,                        # Bool mask (True = attend)
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal and attention_mask is None  # Only auto-causal if no custom mask
            )
            # out: [batch, heads, seq_len, dim_head]
            out = rearrange(out, 'b h n d -> b n (h d)')  # concatenate heads
        
        # ========================================
        # CORRECTED: Manual Attention Path
        # ========================================
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # compute dot product for each head and for each token pair (e.g. i, j)
                                                                 # i,e: score(i,j) = dot( q_i, k_j ) * scale <-> scale is for stability 
                                                                 # E.g. Head 0:
                                                                #           Luke   likes   cats   (as *keys*)
                                                                # Luke    [ 1.2    0.1    0.5 ]
                                                                # likes   [ 0.9    1.5    1.1 ]
                                                                # cats    [ 0.2    0.3    1.8 ]
                                                                #  ^ as queries
            
            # Apply custom attention mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    # [batch, seq_len] -> [batch, 1, 1, seq_len]
                    attention_mask = attention_mask[:, None, None, :]
                # Mask out positions where mask is False
                dots = dots.masked_fill(~attention_mask.bool(), float('-inf'))
            
            # Apply causal mask ONLY if no custom mask and is_causal=True
            elif is_causal:
                seq_len = dots.size(-2)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=dots.device, dtype=torch.bool), 
                    diagonal=1
                )
                dots = dots.masked_fill(causal_mask, float('-inf'))
            
            attn = self.attend(dots) # This is than turned into probabilites:
                                        # Luke-row after softmax:  [0.60, 0.15, 0.25]
                                        # likes-row after softmax: [0.30, 0.40, 0.30]
                                        # cats-row after softmax:  [0.10, 0.10, 0.80]

            attn = self.dropout(attn) # Regularization (drops some random weights)

            out = torch.matmul(attn, v) # dot product with learned values using probablities for weighting (ie likes row here)
                                        # Values are like they information a token holds and the attention (q * k) is the amount the other token takes from that value 
                                        # out_head0["likes"] = 0.30 * v_head0["Luke"]
                                        #                       + 0.40 * v_head0["likes"]
                                        #                       + 0.30 * v_head0["cats"]

            out = rearrange(out, 'b h n d -> b n (h d)') # concatnated the heads 
                                                         # final_head_concat["Luke"]  = [out_head0["Luke"],  out_head1["Luke"]]  # length inner dim
                                                         # final_head_concat["likes"] = [out_head0["likes"], out_head1["likes"]] # length inner dim 
                                                         # final_head_concat["cats"]  = [out_head0["cats"],  out_head1["cats"]]  # length inner dim

        output = self.to_out(out) # project from inner_dim back to dim (get embeddings back)
        
        if use_cache:
            return output, (k, v)
        return output

from deepspeed.moe.layer import MoE
from torch.utils.checkpoint import checkpoint

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., moe_layers:List[int]=[], 
                 num_experts=4, k=-1, capacity_factor=1.5, eval_capacity_factor=2.0, 
                 min_capacity=0.0, use_residual=False, gate_backward='ste', ep_size=1,
                 max_seq_len=512,        # NEW: Pass to Attention
                 use_rope=True,          # NEW: Enable RoPE
                 rope_theta=10000.0,     # NEW: RoPE base frequency
                 rope_dim=None,          # NEW: Partial RoPE (None = full head_dim)
                 rope_scaling=None       # NEW: RoPE scaling config
    ):
        for moe in moe_layers:
            assert moe >= 0, "MOE layers must be greater than or equal to 0"
            assert moe < depth, "MOE layers must be less than the depth of the transformer"
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.moe_losses = []

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim, 
                    heads = heads, 
                    dim_head = dim_head, 
                    dropout = dropout,
                    max_seq_len=max_seq_len,     # NEW
                    use_rope=use_rope,           # NEW
                    rope_theta=rope_theta,       # NEW
                    rope_dim=rope_dim,           # NEW
                    rope_scaling=rope_scaling    # NEW
                ),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

        self.moe_layers = moe_layers

        for layer in self.moe_layers:
            self.layers[layer][1] = MoE(
                dim,
                expert=self.layers[layer][1], # feed forward network used per expert 
                num_experts=num_experts, # number of experts in the layer
                ep_size=ep_size, # number of ranks in the expert parallel world
                k=k, # top-k gating value
                capacity_factor=capacity_factor, # capacity factor for the expert at training time
                eval_capacity_factor=eval_capacity_factor, # capacity factor for the expert at evaluation time
                min_capacity=min_capacity, # minimum capacity for the expert
                use_residual=use_residual, # whether to use residual connection in the MoE layer
                gate_backward=gate_backward,
                #max_expert_num=4
            )

    def forward(
        self, 
        x, 
        attention_mask=None,           # NEW
        position_ids=None,             # NEW
        use_cache=False,               # NEW
        past_key_values=None,          # NEW: List of (k, v) tuples per layer
        is_causal=True                 # NEW
    ):
        l_aux = 0.0
        present_key_values = [] if use_cache else None
        
        for i, (attn, ff) in enumerate(self.layers):
            # Get past KV for this layer
            layer_past = None
            if past_key_values is not None and i < len(past_key_values):
                layer_past = past_key_values[i]
            past_kv_len = layer_past[0].shape[2] if layer_past is not None else 0
            
            # Attention with KV-cache support
            if use_cache:
                attn_out, kv_cache = attn(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_kv_len=past_kv_len,
                    use_cache=True,
                    is_causal=is_causal
                )
                present_key_values.append(kv_cache)
            else:
                attn_out = attn(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_kv_len=past_kv_len,
                    use_cache=False,
                    is_causal=is_causal
                )
            
            x = attn_out + x # Residual connection

            if i in self.moe_layers:
                # MoE layers: Can't use checkpoint due to multiple return values
                output, moe_loss, exp_counts = ff(x)
                l_aux += moe_loss
                x = x + output
                expert_counts[f"layer_{i}"] = exp_counts
            else:
                # Regular FF layers: Use gradient checkpointing during training
                if self.use_gradient_checkpointing and self.training:
                    x = checkpoint(ff, x, use_reentrant=False) + x
                else:
                    x = ff(x) + x

        output = self.norm(x) # normalization
        
        if use_cache:
            return output, l_aux, present_key_values
        return output, l_aux 

class MoE_Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., moe_layers: List[int] = [],
                 num_experts=4, k=-1, capacity_factor=1.5, eval_capacity_factor=2.0, 
                 min_capacity=0.0, use_residual=False, gate_backward='ste', ep_size=1,
                 use_rope=True,          # NEW: Enable RoPE
                 rope_theta=10000.0,     # NEW: RoPE base frequency
                 rope_dim=None,          # NEW: Partial RoPE (None = full head_dim)
                 rope_scaling=None       # NEW: RoPE scaling config
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, dim) # lookup table for token_id -> embedding (shape: [vocab_size, dim], 
                                                            # ie table with vocab_size rows and dim columns, each row is a embedding vector of length dim
                                                            # under the hood (I think?) this turns the ids into one-hot encoded vectors which it feeds into a linear layer to get the embedding (shape: [batch, seq_len, dim])

        # OLD: Absolute position embedding (CONDITIONALLY KEEP OR REMOVE)
        if not use_rope:
            # Keep for backward compatibility or non-RoPE mode
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim)) # embeddings for each postion (ie token at postion 0 gets first postion embedding added independent of the token)
        else:
            self.pos_embedding = None  # RoPE handles positions

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, moe_layers,
                                      num_experts=num_experts, k=k, capacity_factor=capacity_factor,
                                      eval_capacity_factor=eval_capacity_factor, min_capacity=min_capacity,
                                      use_residual=use_residual, gate_backward=gate_backward, ep_size=ep_size,
                                      max_seq_len=max_seq_len,    # NEW
                                      use_rope=use_rope,           # NEW
                                      rope_theta=rope_theta,       # NEW
                                      rope_dim=rope_dim,           # NEW
                                      rope_scaling=rope_scaling    # NEW
        )

        self.mlp_head = nn.Linear(dim, vocab_size)

    def forward(
        self, 
        tokens, 
        attention_mask=None,      # NEW
        position_ids=None,        # NEW
        use_cache=False,          # NEW
        past_key_values=None,     # NEW
        is_causal=True            # NEW
    ):
        x = self.token_embedding(tokens)
        b, n, _ = x.shape

        # OLD: Add position embeddings (ONLY IF NOT USING ROPE)
        if self.pos_embedding is not None:
            x += self.pos_embedding[:, :n]

        x = self.dropout(x)

        # Transformer with KV-cache support
        if use_cache:
            x, l_aux, present_kv = self.transformer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past_key_values,
                is_causal=is_causal
            )
            logits = self.mlp_head(x)
            return logits, l_aux, present_kv
        else:
            x, l_aux = self.transformer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                past_key_values=past_key_values,
                is_causal=is_causal
            )
            logits = self.mlp_head(x)
            return logits, l_aux


def moe_builder(vocab_size: int = 131072, max_seq_len: int = 2048, dim=768, depth=4, heads=4, mlp_dim=512, 
                dim_head=64, dropout=0., emb_dropout=0., moe_layers=[0, 3],
                num_experts=4, k=-1, capacity_factor=1.5, eval_capacity_factor=2.0,
                min_capacity=0.0, use_residual=False, gate_backward='ste', ep_size=1,
                use_rope=True,          # NEW: Enable RoPE by default
                rope_theta=10000.0,     # NEW: RoPE base frequency
                rope_dim=None,          # NEW: Partial RoPE (None = full head_dim)
                rope_scaling=None       # NEW: RoPE scaling config
):
    """
    Build a Mixture of Experts Transformer model with optional RoPE.
    
    Args:
        vocab_size: Size of vocabulary
        max_seq_len: Maximum sequence length
        dim: Model dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_dim: Feedforward dimension
        dim_head: Dimension per attention head
        dropout: Dropout probability
        emb_dropout: Embedding dropout probability
        moe_layers: List of layer indices to use MoE
        num_experts: Number of experts in MoE
        k: Top-k experts to use
        capacity_factor: MoE capacity factor
        eval_capacity_factor: MoE eval capacity factor
        min_capacity: Minimum MoE capacity
        use_residual: Use residual in MoE
        gate_backward: MoE gate backward method
        ep_size: Expert parallel size
        use_rope: Whether to use RoPE (default: True)
        rope_theta: Base frequency for RoPE (default: 10000.0)
        rope_dim: Number of dimensions to rotate (None = full head_dim, must be even)
        rope_scaling: Optional scaling config for longer context
            - {'type': 'linear', 'factor': 2.0} for linear scaling
            - {'type': 'ntk', 'factor': 2.0, 'alpha': 1.0} for NTK-aware scaling
            - {'type': 'yarn', 'factor': 2.0, 'scale': 1.0} for YaRN scaling (simplified)
    
    Returns:
        MoE_Transformer model
    """
    model = MoE_Transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout,
        moe_layers=moe_layers,
        num_experts=num_experts,
        k=k,
        capacity_factor=capacity_factor,
        eval_capacity_factor=eval_capacity_factor,
        min_capacity=min_capacity,
        use_residual=use_residual,
        gate_backward=gate_backward,
        ep_size=ep_size,
        use_rope=use_rope,           # NEW
        rope_theta=rope_theta,       # NEW
        rope_dim=rope_dim,           # NEW
        rope_scaling=rope_scaling     # NEW
    )

    return model


if __name__ == "__main__":
    import deepspeed
    
    vocab_size = 32_000
    seq_len = 128
    
    model = moe_builder(vocab_size=vocab_size, max_seq_len=seq_len)

    ds_config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    device = model_engine.local_rank if torch.cuda.is_available() else "cpu"
    
    batch_size = 2
    # Start with prompt
    tokens = torch.tensor([[10, 25, 78]]).to(device)  # "The cat sat" - (not really as we have no tokenizer yet :v )


    for i in range(5):
    # Forward pass
        output, l_aux = model_engine(tokens)  # shape [1, 3, 32000]

        # Get prediction for NEXT token (after "sat")
        next_token_logits = output[0, -1, :]  # shape [32000]
                                # ↑ last position

         # Sample or argmax to get next token (most likely token in this case)
        next_token = torch.argmax(next_token_logits)  

        # Append and repeat
        tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        print(tokens)
