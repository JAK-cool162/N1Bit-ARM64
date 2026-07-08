#!/usr/bin/env python3
"""Train the 1-bit model, then package it into a runnable .zip bundle.

    python train.py                     # train DEFAULT_STEPS, then bundle
    python train.py --steps 500         # train 500 steps
    python train.py --steps inf         # train forever (Ctrl-C to stop & bundle)
    python train.py --name coder        # name the output bundle builds/coder.zip
    python train.py --no-zip            # just train, don't build a .zip

Data must exist first:  python -m n1bit.data
"""

import argparse

from n1bit.config import DEFAULT_STEPS
from n1bit.trainer import Trainer
from n1bit.packager import package


def main():
    ap = argparse.ArgumentParser(description="Train N1Bit and export a .zip bundle")
    ap.add_argument("--steps", default=str(DEFAULT_STEPS),
                    help="number of steps, or 'inf' for infinite")
    ap.add_argument("--name", default="n1bit_model", help="bundle name")
    ap.add_argument("--no-zip", action="store_true", help="skip building the .zip")
    args = ap.parse_args()

    trainer = Trainer(on_update=_print_dash)
    steps = "inf" if args.steps == "inf" else int(args.steps)
    print(f"[train] device={trainer.device}  steps={args.steps}")
    try:
        trainer.train(steps)
    except KeyboardInterrupt:
        print("\n[train] stopped by user")
        trainer.save()

    if not args.no_zip:
        path = package(args.name)
        print(f"\n[done] bundle ready: {path}")
        print(f"       run it:  python -c \"import zipfile; zipfile.ZipFile('{path}').extractall('.')\"")
        print(f"                python {args.name}/run.py chat")


_last = [0]


def _print_dash(s):
    print(f"step {s['step']:>6} | loss {s['loss']:.3f} | best {s['best']:.3f} "
          f"| {s['tok_s']:,.0f} tok/s | {s['grade']:<16} | {s['sample'][:60]!r}")


if __name__ == "__main__":
    main()
