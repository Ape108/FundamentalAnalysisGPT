import torch
import math


class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()  # initialize base class

        self.eps = eps  # numerical stability term

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = torch.nn.Parameter(torch.ones(emb_dim))  # [D]
        self.beta = torch.nn.Parameter(torch.zeros(emb_dim))  # [D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        # Compute mean over the feature dimension D
        mean = x.mean(dim=-1, keepdim=True)  # [B, T, 1]

        # Compute variance over the feature dimension D
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, T, 1]

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # [B, T, D]

        # Scale and shift (broadcast gamma/beta over B and T)
        out = self.gamma * x_hat + self.beta  # [B, T, D]

        return out

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim: int, drop_rate: float):
        super().__init__()  # initialize

        # Two-layer MLP with GELU in between
        self.net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),  # expand features
            torch.nn.GELU(),                        # nonlinearity
            torch.nn.Linear(4 * emb_dim, emb_dim),  # project back to emb_dim
            torch.nn.Dropout(drop_rate)             # dropout for regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self.net(x)  # [B, T, D]

class MultiHeadCausalSelfAttention(torch.nn.Module):
    # Multi-head causal self-attention (scaled dot-product)
    def __init__(self, emb_dim: int, num_heads: int, context_length: int, drop_rate: float, qkv_bias: bool):
        super().__init__()  # init

        assert emb_dim % num_heads == 0  # ensure heads divide embedding dim

        self.emb_dim = emb_dim  # embedding dimension D
        self.num_heads = num_heads  # number of heads H
        self.head_dim = emb_dim // num_heads  # per-head dim

        # Linear projections for Q, K, V (each produces D features)
        self.Wq = torch.nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wk = torch.nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wv = torch.nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        # Output projection back to emb_dim
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim, bias=True)

        # Dropout on attention weights
        self.attn_drop = torch.nn.Dropout(drop_rate)

        # Register a causal mask as a non-parameter buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape  # unpack

        # Project to Q, K, V: [B, T, D]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Reshape into heads: [B, T, D] -> [B, H, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scores: [B, H, T, T]
        scores = Q @ K.transpose(-2, -1)

        # Scale scores for stability
        scores = scores / math.sqrt(self.head_dim)

        # Apply causal mask (slice to current T)
        mask = self.mask[:T, :T]
        scores = scores.masked_fill(mask, -torch.inf)

        # Softmax over last dim to get attention weights
        weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        weights = self.attn_drop(weights)

        # Context per head: [B, H, T, head_dim]
        context = weights @ V

        # Recombine heads: [B, H, T, head_dim] -> [B, T, D]
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        # Final projection: [B, T, D]
        out = self.out_proj(context)

        return out

class TransformerBlock(torch.nn.Module):
    # GPT-2 style Pre-LN transformer block: (Attn + FFN) with residual connections
    def __init__(self, cfg):
        super().__init__()  # init

        D = cfg["emb_dim"]  # embedding dim

        # Pre-LN layers
        self.ln1 = LayerNorm(D)
        self.ln2 = LayerNorm(D)

        # Causal multi-head attention
        self.attn = MultiHeadCausalSelfAttention(
            emb_dim=D,
            num_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            drop_rate=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Feed-forward network
        self.ff = FeedForward(D, cfg["drop_rate"])

        # Dropout on residual branches (common in GPT-style)
        self.resid_drop = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        # Attention block (Pre-LN) + residual
        x = x + self.resid_drop(self.attn(self.ln1(x)))

        # Feed-forward block (Pre-LN) + residual
        x = x + self.resid_drop(self.ff(self.ln2(x)))

        return x

class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()  # init

        self.cfg = cfg  # store cfg

        # Embeddings
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.blocks = torch.nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Final normalization
        self.final_ln = LayerNorm(cfg["emb_dim"])

        # Output head (logits over vocab)
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        B, T = idx.shape  # unpack

        # Token embeddings: [B, T, D]
        tok = self.tok_emb(idx)

        # Positional embeddings: [T, D]
        pos_ids = torch.arange(T, device=idx.device)
        pos = self.pos_emb(pos_ids)

        # Combine + dropout: [B, T, D]
        x = self.drop_emb(tok + pos)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.final_ln(x)

        # Output logits: [B, T, V]
        logits = self.out_head(x)

        return logits