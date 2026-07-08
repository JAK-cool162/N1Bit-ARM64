"""Training loop: streams cache/data.bin, autosaves, resumes, and reports progress.

Kept deliberately small — one loop, one checkpoint file.
"""

import os
import json
import time
import numpy as np
import torch

from .config import (DATA_BIN, CHECKPOINT, BATCH_SIZE, SEQ_LEN, LR,
                     SAVE_EVERY, LOG_EVERY)
from .model import N1BitLM, model_from_config, build_config


def grade(loss: float) -> str:
    if loss < 0.8:  return "S - fluent"
    if loss < 1.2:  return "A - coherent"
    if loss < 1.7:  return "B - learning"
    if loss < 2.3:  return "C - noisy"
    if loss < 3.0:  return "D - gibberish"
    return "F - just started"


class Trainer:
    def __init__(self, on_update=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.on_update = on_update  # optional callback(state) for dashboards
        self.state = {"step": 0, "loss": 0.0, "best": float("inf"),
                      "grade": "F", "sample": "(waiting)", "tok_s": 0.0}

        if not os.path.exists(DATA_BIN) or os.path.getsize(DATA_BIN) == 0:
            raise SystemExit("No cache/data.bin — run:  python -m n1bit.data")
        self.data = np.memmap(DATA_BIN, dtype=np.uint16, mode="r")

        self.cfg = build_config()
        self.model = model_from_config(self.cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR)

        if os.path.exists(CHECKPOINT):
            ck = torch.load(CHECKPOINT, map_location=self.device)
            if ck.get("config") == self.cfg:
                self.model.load_state_dict(ck["model"])
                self.opt.load_state_dict(ck["opt"])
                self.state["step"] = ck.get("step", 0)
                self.state["best"] = ck.get("best", float("inf"))
                print(f"[train] resumed at step {self.state['step']}")

    def _batch(self):
        n = len(self.data)
        ix = np.random.randint(0, n - SEQ_LEN - 1, BATCH_SIZE)
        x = np.stack([self.data[i:i + SEQ_LEN] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + SEQ_LEN + 1] for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device))

    def save(self):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "step": self.state["step"], "best": self.state["best"],
                    "config": self.cfg}, CHECKPOINT)

    def train(self, steps, stop_flag=None):
        infinite = steps == "inf"
        limit = float("inf") if infinite else int(steps)
        start_step = self.state["step"]
        self.model.train()
        t0, seen = time.time(), 0

        while self.state["step"] - start_step < limit:
            if stop_flag is not None and stop_flag():
                break
            x, y = self._batch()
            _, loss = self.model(x, y)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            self.state["step"] += 1
            seen += BATCH_SIZE * SEQ_LEN
            lv = loss.item()
            if lv < self.state["best"]:
                self.state["best"] = lv

            if self.state["step"] % LOG_EVERY == 0:
                self.state["loss"] = lv
                self.state["grade"] = grade(lv)
                self.state["tok_s"] = seen / max(time.time() - t0, 1e-6)
                t0, seen = time.time(), 0
                self.state["sample"] = self.sample()
                self.model.train()
                if self.on_update:
                    self.on_update(self.state)

            if self.state["step"] % SAVE_EVERY == 0:
                self.save()

        self.save()
        return self.state

    @torch.no_grad()
    def sample(self, prompt="The ", n=80):
        from .tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        ids = torch.tensor([tok.encode(prompt)], device=self.device)
        out = self.model.generate(ids, max_new_tokens=n, temperature=0.8)
        return tok.decode(out[0].tolist())
