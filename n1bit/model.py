"""A tiny 1-bit (BitNet-style) transformer language model.

ONE implementation. The "1-bit" part lives in `BitLinear`, which quantizes its
weights to {-1, +1} on every forward pass and uses a straight-through estimator so
gradients still flow. Everything else is a normal small GPT-style decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, SEQ_LEN, EOS_ID


class BitLinear(nn.Module):
    """Linear layer whose weights are binarized to {-1, +1} * scale.

    Uses the straight-through estimator (STE): forward pass uses the binarized
    weights, backward pass pretends the binarization was the identity so the
    full-precision shadow weights keep learning.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-5)
        w_bin = torch.sign(w)
        w_bin = torch.where(w_bin == 0, torch.ones_like(w_bin), w_bin) * scale
        # STE: value of w_bin, gradient of w
        w_ste = w + (w_bin - w).detach()
        return F.linear(x, w_ste, self.bias)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = BitLinear(dim, dim * 3)
        self.proj = BitLinear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = BitLinear(dim, dim * 4)
        self.fc2 = BitLinear(dim * 4, dim)

    def forward(self, x, mask):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class N1BitLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=EMBED_DIM, layers=NUM_LAYERS,
                 heads=NUM_HEADS, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = BitLinear(dim, vocab_size)
        self.register_buffer(
            "mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x, self.mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == EOS_ID:
                break
        return idx


def build_config() -> dict:
    """The architecture description saved with every checkpoint / bundle."""
    return {
        "vocab_size": VOCAB_SIZE,
        "dim": EMBED_DIM,
        "layers": NUM_LAYERS,
        "heads": NUM_HEADS,
        "seq_len": SEQ_LEN,
    }


def model_from_config(cfg: dict) -> "N1BitLM":
    return N1BitLM(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        seq_len=cfg["seq_len"],
    )
