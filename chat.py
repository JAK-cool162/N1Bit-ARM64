#!/usr/bin/env python3
"""Terminal dashboard + chat for a trained N1Bit model.

    python chat.py           # chat with the current checkpoint
    python chat.py --dash    # live training dashboard (trains + shows progress)

Loads cache/model.pt (the checkpoint produced by train.py).
"""

import os
import argparse
import torch

from n1bit.config import CHECKPOINT
from n1bit.model import model_from_config
from n1bit.tokenizer import ByteTokenizer


def load_model():
    if not os.path.exists(CHECKPOINT):
        raise SystemExit("No checkpoint — run:  python train.py")
    ck = torch.load(CHECKPOINT, map_location="cpu")
    m = model_from_config(ck["config"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def chat():
    m, ck = load_model()
    tok = ByteTokenizer()
    params = sum(p.numel() for p in m.parameters())
    print("=" * 50)
    print(f"  N1Bit chat  |  {params:,} params  |  step {ck.get('step', 0)}")
    print(f"  best loss {ck.get('best', float('inf')):.3f}  |  Ctrl-C to quit")
    print("=" * 50)
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user:
            continue
        ids = torch.tensor([tok.encode(user + "\n")])
        out = m.generate(ids, max_new_tokens=200, temperature=0.8)
        print("ai >", tok.decode(out[0].tolist()))


def dashboard():
    from n1bit.trainer import Trainer

    def render(s):
        os.system("clear" if os.name != "nt" else "cls")
        print("=" * 52)
        print("   N1Bit LIVE TRAINING DASHBOARD")
        print("=" * 52)
        print(f"   Step:   {s['step']:,}")
        print(f"   Loss:   {s['loss']:.4f}   Best: {s['best']:.4f}")
        print(f"   Speed:  {s['tok_s']:,.0f} tokens/sec")
        print(f"   Grade:  {s['grade']}")
        print("-" * 52)
        print(f"   Sample: {s['sample'][:200]}")
        print("=" * 52)
        print("   Ctrl-C to stop and save")

    t = Trainer(on_update=render)
    try:
        t.train("inf")
    except KeyboardInterrupt:
        t.save()
        print("\nsaved. run `python chat.py` to talk to it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dash", action="store_true", help="live training dashboard")
    args = ap.parse_args()
    dashboard() if args.dash else chat()


if __name__ == "__main__":
    main()
