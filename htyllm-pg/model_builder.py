import token
import torch
from torch import dtype, nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.cpu import is_available


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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

from deepspeed.moe.layer import MoE
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.moe_losses = []

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.moe_layers = [0, 3]

        for layer in self.moe_layers:
            self.layers[layer][1] = MoE(
                dim,
                expert=self.layers[layer][1],
                num_experts=2,
                ep_size=1,
                k=1,
                capacity_factor=1.5,
                eval_capacity_factor=2.0,
                min_capacity=0.0,
                use_residual=False,
                # max_expert_num=4
            )

    def forward(self, x):

        l_aux = 0.0
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x

            if i in self.moe_layers:
                output, moe_loss, _ = ff(x)
                l_aux += moe_loss
                x = x + output
            else:
                x = ff(x) + x

        return self.norm(x), l_aux

class MoE_Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, dim) # lookup table for token_id -> embedding (shape: [vocab_size, dim], 
                                                            # ie table with vocab_size rows and dim columns, each row is a embedding vector of length dim

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim)) # embeddings for each postion (ie token at postion 0 gets first postion embedding added independent of the token)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, vocab_size)

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x, l_aux = self.transformer(x)
        return self.mlp_head(x), l_aux


def moe_builder():
    model = MoE_Transformer(
        vocab_size=32_000,
        max_seq_len=500,
        dim=768,
        depth=4,
        heads=4,
        mlp_dim=512
    )

    return model


if __name__ == "__main__":
    import deepspeed
    model = moe_builder()

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
    
    vocab_size = 32_000
    batch_size = 2
    seq_len = 128
    # Start with prompt
    tokens = torch.tensor([[10, 25, 78]]).to(device)  # "The cat sat"

    # Forward pass
    output, l_aux = model_engine(tokens)  # shape [1, 3, 32000]

    # Get prediction for NEXT token (after "sat")
    next_token_logits = output[0, -1, :]  # shape [32000]
                                # ↑ last position

    # Sample or argmax to get next token
    next_token = torch.argmax(next_token_logits)  # e.g., token 92 = "on"

    # Append and repeat
    tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    tokens = torch.cat([tokens, tokens],dim=0)
    print(tokens)
    output,l_aux=model_engine(tokens)
    print(output)
    # Now tokens = [[10, 25, 78, 92]] → "The cat sat on"
