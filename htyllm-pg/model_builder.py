from typing import List
import torch
from torch import nn

from einops import rearrange


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

        self.heads = heads # number of attention heads (how many attentions are stacked per layer)
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
        x = self.norm(x)# nomralize each tokens along dimension (mean 0, variance 1) shape stays 

        qkv = self.to_qkv(x).chunk(3, dim = -1)# linear layer that maps each 8-dim vector to a big vector (inner_dim * 3)
                                               # 3 because after chunk the is one for q, k, v (shape each: (batch, tokens, inner_dim) )
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # split into multiple heads 
                                                                                                # query, key, and values for each token for each header
                                                                                                # Heads might focus on different things in a sentence (syntax, semantic, ...who knows:v)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # compute dot product for each head and for each token pair (e.g. i, j)
                                                                 # i,e: score(i,j) = dot( q_i, k_j ) * scale <-> scale is for stability 
                                                                 # E.g. Head 0:
                                                                #           Luke   likes   cats   (as *keys*)
                                                                # Luke    [ 1.2    0.1    0.5 ]
                                                                # likes   [ 0.9    1.5    1.1 ]
                                                                # cats    [ 0.2    0.3    1.8 ]
                                                                #  ^ as queries
        attn = self.attend(dots) # This is than turned into probabilites:
                                    # Luke-row after softmax:  [0.60, 0.15, 0.25]
                                    # likes-row after softmax: [0.30, 0.40, 0.30]
                                    # cats-row after softmax:  [0.10, 0.10, 0.80]

        attn = self.dropout(attn) # Regularization (drops some random weights)

        out = torch.matmul(attn, v) # dot product with learned values using probablities for weighting (ie likes row here)
                                    # out_head0["likes"] = 0.30 * v_head0["Luke"]
                                    #                       + 0.40 * v_head0["likes"]
                                    #                       + 0.30 * v_head0["cats"]

        out = rearrange(out, 'b h n d -> b n (h d)') # concatnated the heads 
                                                     # final_head_concat["Luke"]  = [out_head0["Luke"],  out_head1["Luke"]]  # length inner dim
                                                     # final_head_concat["likes"] = [out_head0["likes"], out_head1["likes"]] # length inner dim 
                                                     # final_head_concat["cats"]  = [out_head0["cats"],  out_head1["cats"]]  # length inner dim

        return self.to_out(out) # project from inner_dim back to dim (get embeddings back)

from deepspeed.moe.layer import MoE
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., moe_layers:List[int]=[]):
        for moe in moe_layers:
            assert moe >= 0, "MOE layers must be greater than or equal to 0"
            assert moe < depth, "MOE layers must be less than the depth of the transformer"
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.moe_losses = []

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.moe_layers = moe_layers

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
    def __init__(self, vocab_size, max_seq_len, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., moe_layers: List[int] = []):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, dim) # lookup table for token_id -> embedding (shape: [vocab_size, dim], 
                                                            # ie table with vocab_size rows and dim columns, each row is a embedding vector of length dim

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim)) # embeddings for each postion (ie token at postion 0 gets first postion embedding added independent of the token)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, moe_layers)

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
        mlp_dim=512,
        moe_layers=[0, 3]
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
