import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
import argparse
import random
import sentencepiece as spm


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 24 # number of layers
    n_head: int = 16 # number of heads
    n_embd: int = 1024 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() > 1]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() <= 1]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"number of decay params: {num_decay_params} in number of tensors: {len(decay_params)}")
        print(f"number of no decay params: {num_no_decay_params} in number of tensors: {len(no_decay_params)}")
        use_fused = device == 'cuda'
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
# -----------------------------------------------------------------------------
import numpy as np

def load_tokens(filename):
    print(f"DEBUG: Loading tokens from {filename}", flush=True)
    try:
        npt = np.load(filename)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        print(f"DEBUG: Successfully loaded tokens, shape: {ptt.shape}", flush=True)
        return ptt
    except Exception as e:
        print(f"ERROR: Failed to load tokens from {filename}: {e}", flush=True)
        raise

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_path):
        print(f"DEBUG: Initializing DataLoaderLite with rank={process_rank}, processes={num_processes}, split={split}", flush=True)
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = os.path.abspath(data_path)
        print(f"DEBUG: Looking for data in {data_root}", flush=True)
        
        try:
            shards = os.listdir(data_root)
            print(f"DEBUG: Found {len(shards)} total files in data directory", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to list directory {data_root}: {e}", flush=True)
            raise

        # Filter to only include .npy files and sort them
        shards = [s for s in shards if s.endswith('.npy')]
        shards = sorted(shards)
        total_shards = len(shards)
        split_idx = int(total_shards * 0.9)  # 90% for training
        
        if split == 'train':
            shards = shards[:split_idx]
        else:  # val
            shards = shards[split_idx:]
            
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
            print(f"DEBUG: First few shards: {shards[:3]}", flush=True)
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train GPT-2 model')
    parser.add_argument('--data_path', type=str, default="../data/edu_fineweb10B",
                        help='Path to the data directory containing training shards')
    parser.add_argument('--tokenizer_path', type=str, default="tokenizer/sp_model.model",
                        help='Path to the SentencePiece tokenizer model file')
    parser.add_argument('--batch_size', type=int, default=524288,
                        help='Total batch size for training')
    parser.add_argument('--micro_batch', type=int, default=32,
                        help='Micro batch size')
    parser.add_argument('--seq_len', type=int, default=1024,
                        help='Sequence length')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Save a checkpoint every N steps')
    args = parser.parse_args()
    
    import torch.distributed as dist
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this ddp run?

    if ddp:
        ddp_rank = int(os.environ.get('RANK'))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
        ddp_world_size = int(os.environ.get('WORLD_SIZE'))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
        # Initialize process group
        init_process_group(backend='nccl')
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")


    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Load SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_path)
    if master_process:
        print(f"Loaded SentencePiece tokenizer from {args.tokenizer_path}")
        print(f"Vocabulary size: {sp.get_piece_size()}")

    total_batch_size = args.batch_size
    B = args.micro_batch  # "micro" batch size
    T = args.seq_len  # sequence length
    assert total_batch_size % B * T * ddp_world_size == 0, f"total batch size {total_batch_size} must be divisible by B*T*ddp_world_size {B*T*ddp_world_size}"
    grad_acum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size} = {B} * {T} * {grad_acum_steps}") 

    import sys; 
    print("This is GPU ", ddp_rank)

    torch.set_float32_matmul_precision('high')

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', data_path=args.data_path) # max batch size depends on your GPU memory (should be power of 2)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_path=args.data_path)

    model = GPT(GPTConfig(vocab_size=sp.get_piece_size()))  # Use SentencePiece vocab size
    model.to(device)
    model = torch.compile(model)
    if ddp: 
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # unwrapped model

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 1024  # scaled values according to gpt-2 paper (we use other dataset)
    max_steps = 32_768 # scaled values according to gpt-2 paper (we use other dataset)

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it >max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    import time
    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

    # Create directory for checkpoints
    checkpoint_dir = args.checkpoint_dir
    if master_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # Define checkpoint frequency
    checkpoint_interval = args.checkpoint_interval

    if master_process:
        print("training...")
    for step in range(max_steps):
        t0 = time.time()
        # validation
        if step % 100 == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"step {step+1}/{max_steps}; val loss {val_loss_accum:.4f}")
            model.train()
        
        # Save checkpoint
        if step > 0 and step % checkpoint_interval == 0 and master_process:
            checkpoint_path = os.path.join(checkpoint_dir, f"gpt2_model_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_accum.item() if 'val_loss_accum' in locals() else None,
            }, checkpoint_path)
            print(f"Saved checkpoint at step {step} to {checkpoint_path}")
        
        # generate from model
        if step > 0 and step % 100 == 0:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            
            # Define 10 English and 10 German sentence beginnings
            english_prompts = [
                "What happened on the ",
                "The most important thing is ",
                "I would like to discuss ",
                "Once upon a time there was ",
                "The best way to learn is ",
                "Scientists have discovered that ",
                "When I look at the stars ",
                "The history of the world ",
                "Technology has changed how we ",
                "In the beginning there was "
            ]
            
            german_prompts = [
                "Eines Tages werde ich ",
                "Die wichtigste Sache ist ",
                "Ich möchte gerne über ",
                "Es war einmal ein ",
                "Der beste Weg zu lernen ist ",
                "Wissenschaftler haben entdeckt, dass ",
                "Wenn ich die Sterne betrachte ",
                "Die Geschichte der Welt ",
                "Technologie hat verändert, wie wir ",
                "Am Anfang war "
            ]
            
            # Combine all prompts into one list
            all_prompts = english_prompts + german_prompts
            
            # Randomly select one prompt
            random.seed(step + ddp_rank)  # Ensure reproducibility but different per step and rank
            selected_prompt = random.choice(all_prompts)
            
            # Encode the selected prompt using SentencePiece
            tokens = sp.encode(selected_prompt)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            
            # Track which type of prompt was selected for logging
            prompt_type = "English" if selected_prompt in english_prompts else "German"
            
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(xgen)
                    logits = logits[:, -1, :] # (B, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    xgen = torch.cat((xgen, xcol), dim=1)
                    
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = sp.decode(tokens)  # Use SentencePiece to decode
                print(f"rank {ddp_rank}, sample {i}, prompt type: {prompt_type}: {decoded}")
                    
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_acum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_acum_steps
            loss_accum += loss.detach()
            if ddp: 
                model.require_backward_grad_sync = (micro_step == grad_acum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        t1 = time.time()
        dt =  (t1 - t0) 
        tokens_processed = B * T * grad_acum_steps * ddp_world_size
        tokens_per_second = tokens_processed / dt
        if master_process:
            print(f"step {step+1}/{max_steps}; loss {loss_accum.item():.4f}; norm {norm:.2f}; lr {lr:.6f}; "
                f"tokens/sec {tokens_per_second:.2f}; dt {dt:.2f}ms")

    # Save the final model after training completes
    if master_process:
        final_model_path = f"gpt2_fineweb2_model_steps_{max_steps}.pt"
        model_to_save = raw_model
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod

        torch.save({
            'step': step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_model_path)

        print(f"Final model saved to {final_model_path}")

    if ddp:
        destroy_process_group()
