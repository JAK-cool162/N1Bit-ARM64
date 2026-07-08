"""Package a finished model into a runnable .zip bundle.

Instead of a .gguf, training produces  builds/<name>.zip  containing:

    <name>/
      model.pt        # weights + architecture config
      data.bin        # (optional) a slice of the training data, for context/finetune
      tokenizer.py    # the byte tokenizer (self-contained copy)
      n1bit_mini.py   # tiny self-contained model definition (no repo needed)
      run.py          # the CLI: chat / sample / info / sandbox
      README.txt      # how to run it

The bundle is self-contained: unzip it anywhere with PyTorch installed and run
    python <name>/run.py chat
No need for the rest of the repo.
"""

import os
import json
import shutil
import zipfile
import tempfile
import inspect

import torch

from .config import CHECKPOINT, DATA_BIN, BUILDS_DIR
from . import model as model_mod


RUNNER = r'''#!/usr/bin/env python3
"""Self-contained runner for an N1Bit bundle.

Commands:
    python run.py chat            # interactive chat
    python run.py sample "text"   # continue a prompt once
    python run.py info            # model stats
    python run.py sandbox         # (extensible) run AI output in a scratch dir

Only needs: torch.
"""
import os, sys, json, subprocess, tempfile
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n1bit_mini import N1BitLM
from tokenizer import ByteTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
tok = ByteTokenizer()


def load():
    ck = torch.load(os.path.join(HERE, "model.pt"), map_location="cpu")
    cfg = ck["config"]
    m = N1BitLM(cfg["vocab_size"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["seq_len"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def generate(m, prompt, n=200, temp=0.8):
    ids = torch.tensor([tok.encode(prompt)])
    out = m.generate(ids, max_new_tokens=n, temperature=temp)
    return tok.decode(out[0].tolist())


def cmd_chat():
    m, _ = load()
    print("N1Bit chat — Ctrl-C to quit\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        print("ai>", generate(m, user + "\n", n=200))


def cmd_sample():
    m, _ = load()
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The "
    print(generate(m, prompt, n=300))


def cmd_info():
    m, ck = load()
    params = sum(p.numel() for p in m.parameters())
    print(json.dumps({"config": ck["config"], "step": ck.get("step"),
                      "best_loss": ck.get("best"), "parameters": params}, indent=2))


def cmd_sandbox():
    """Ask the model for code, then run it in a throwaway directory.

    This is the extension point for 'sandbox AI' and future tools. It is opt-in
    and runs plain Python in a temp dir — extend with proot/containers as needed.
    """
    m, _ = load()
    task = " ".join(sys.argv[2:]) or input("task> ")
    code = generate(m, "# python\n" + task + "\n", n=300)
    print("--- generated ---\n" + code + "\n-----------------")
    if input("run this? [y/N] ").strip().lower() != "y":
        return
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gen.py")
        open(p, "w").write(code)
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=20)
        print("stdout:\n" + r.stdout)
        print("stderr:\n" + r.stderr)
        print("exit:", r.returncode)


CMDS = {"chat": cmd_chat, "sample": cmd_sample, "info": cmd_info, "sandbox": cmd_sandbox}

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "chat"
    CMDS.get(c, cmd_chat)()
'''

README_TXT = """N1Bit bundle: {name}
=====================

A tiny 1-bit language model, fully self-contained. Requires Python + PyTorch.

    pip install torch
    python {name}/run.py chat            # talk to it
    python {name}/run.py sample "Once"   # continue a prompt
    python {name}/run.py info            # show model stats
    python {name}/run.py sandbox "..."   # generate + run code (extensible)

Files:
    model.pt       weights + architecture
    data.bin       sample of training data (optional context / finetune)
    tokenizer.py   byte tokenizer
    n1bit_mini.py  self-contained model definition
    run.py         the CLI above
"""

# a self-contained copy of the model definition (no imports from the package)
MINI_MODEL = '''"""Self-contained N1Bit model (no external package needed)."""
import math, torch, torch.nn as nn, torch.nn.functional as F


class BitLinear(nn.Module):
    def __init__(self, i, o, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(o, i)); nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(o)) if bias else None
    def forward(self, x):
        w = self.weight; s = w.abs().mean().clamp(min=1e-5)
        wb = torch.sign(w); wb = torch.where(wb == 0, torch.ones_like(wb), wb) * s
        return F.linear(x, w + (wb - w).detach(), self.bias)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h; self.hd = d // h
        self.ln1 = nn.LayerNorm(d); self.qkv = BitLinear(d, d*3); self.proj = BitLinear(d, d)
        self.ln2 = nn.LayerNorm(d); self.fc1 = BitLinear(d, d*4); self.fc2 = BitLinear(d*4, d)
    def forward(self, x, mask):
        B, T, C = x.shape; hs = self.ln1(x)
        q, k, v = self.qkv(hs).split(C, dim=2)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2); k = k.view(B, T, self.h, self.hd).transpose(1, 2); v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        a = a.masked_fill(mask[:, :, :T, :T] == 0, float("-inf")); a = F.softmax(a, dim=-1)
        y = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y); return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class N1BitLM(nn.Module):
    def __init__(self, vocab_size, dim, layers, heads, seq_len):
        super().__init__(); self.seq_len = seq_len; self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, dim); self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim); self.head = BitLinear(dim, vocab_size)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
    def forward(self, idx, targets=None):
        B, T = idx.shape; pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks: x = b(x, self.mask)
        logits = self.head(self.ln_f(x)); loss = None
        if targets is not None: loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.seq_len:]); logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))); logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1); idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == 258: break
        return idx
'''


def package(name: str = "n1bit_model", include_data_mb: float = 5.0) -> str:
    """Create builds/<name>.zip from the current checkpoint. Returns the zip path."""
    if not os.path.exists(CHECKPOINT):
        raise SystemExit("No checkpoint to package — train first.")

    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, name)
    os.makedirs(root, exist_ok=True)

    shutil.copy(CHECKPOINT, os.path.join(root, "model.pt"))

    # a small slice of data.bin for context / future finetuning
    if os.path.exists(DATA_BIN):
        cap = int(include_data_mb * 1_000_000)
        with open(DATA_BIN, "rb") as src, open(os.path.join(root, "data.bin"), "wb") as dst:
            dst.write(src.read(cap))

    # self-contained tokenizer copy
    open(os.path.join(root, "tokenizer.py"), "w").write(_tokenizer_source())
    open(os.path.join(root, "n1bit_mini.py"), "w").write(MINI_MODEL)
    open(os.path.join(root, "run.py"), "w").write(RUNNER)
    open(os.path.join(root, "README.txt"), "w").write(README_TXT.format(name=name))

    zip_path = os.path.join(BUILDS_DIR, f"{name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(root):
            for fn in files:
                full = os.path.join(folder, fn)
                z.write(full, os.path.relpath(full, tmp))
    shutil.rmtree(tmp)
    return zip_path


def _tokenizer_source() -> str:
    """Produce a standalone tokenizer.py with the special-token ids inlined."""
    from .config import PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE
    return f'''"""Self-contained byte tokenizer."""
PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE = {PAD_ID}, {BOS_ID}, {EOS_ID}, {VOCAB_SIZE}


class ByteTokenizer:
    vocab_size = VOCAB_SIZE
    pad_id, bos_id, eos_id = PAD_ID, BOS_ID, EOS_ID

    def encode(self, text, add_eos=False):
        ids = list(text.encode("utf-8", errors="ignore"))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i < 256:
                out.append(i)
        return out.decode("utf-8", errors="ignore")
'''
