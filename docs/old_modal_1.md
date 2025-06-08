def tiny_mixtral_config(
    vocab_size: int = 131_072,
    d_model: int = 384,
    d_ff: int = 1_536,
    n_layers: int = 12,
    n_heads: int = 6,
    n_kv: int = 2,
    n_experts: int = 7,
    top_k: int = 2,
    max_pos: int = 8_192,
    router_aux_coef: float = 1e-3,
):