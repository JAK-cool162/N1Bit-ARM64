# N1Bit-ARM64

A tiny **1-bit (BitNet-style) language model** you can train on a phone, a Termux
shell, or any laptop — then export as a single **runnable `.zip` bundle**.

It's intentionally small and readable: **one PyTorch model**, a **byte-level
tokenizer**, **one training file** (`data.bin`), and a training run that ends by
packaging your model into something you can hand to anyone and run with a couple of
terminal commands.

```
train  ->  cache/model.pt  ->  builds/<name>.zip  ->  python <name>/run.py chat
```

---

## What makes it "1-bit"

The model is a small GPT-style decoder, but every linear layer is a `BitLinear`
(`n1bit/model.py`): on each forward pass the weights are quantized to `{-1, +1}`
(times a learned scale), and a **straight-through estimator** lets gradients keep
training the full-precision shadow weights. That's the BitNet idea, kept to the
essentials.

---

## Install

```bash
pip install -r requirements.txt
```

That's `torch`, `numpy`, and `flask`. (Add `pyarrow` **only** if you want to ingest
Hugging Face `.parquet` datasets.)

---

## 1. Build the dataset — `data.bin`

The dataset builder reads `links.txt` (Hugging Face dataset URLs), asks the HF API
which text file in each repo is smallest, and downloads **smallest → largest** so
you get usable data fast. Everything is cleaned and appended into **one file**,
`cache/data.bin`. Finished datasets are recorded in `cache/manifest.json` and
**skipped next time**, so it never re-downloads what you already have.

```bash
python -m n1bit.data              # download all, smallest first (resumable)
python -m n1bit.data --max 5      # just the 5 smallest, to get started quickly
python -m n1bit.data --reset      # wipe data.bin + manifest and start fresh
```

You can stop after a few small datasets and start training immediately.

---

## 2. Train — and get a `.zip` bundle

```bash
python train.py                   # train, then package builds/n1bit_model.zip
python train.py --steps 500       # train exactly 500 steps
python train.py --steps inf       # train forever (Ctrl-C stops and still bundles)
python train.py --name coder      # name the bundle -> builds/coder.zip
python train.py --no-zip          # train only, skip the bundle
```

Training auto-resumes from `cache/model.pt` and autosaves as it goes.

### The `.zip` bundle (instead of `.gguf`)

When training finishes you get `builds/<name>.zip`. It is **self-contained** — unzip
it anywhere that has Python + PyTorch and run it directly, no repo needed:

```
<name>/
  model.pt        weights + architecture
  data.bin        a slice of training data (context / future finetuning)
  tokenizer.py    the byte tokenizer
  n1bit_mini.py   a standalone copy of the model definition
  run.py          the CLI below
  README.txt      quick instructions
```

```bash
python <name>/run.py chat            # interactive chat
python <name>/run.py sample "Once"   # continue a prompt
python <name>/run.py info            # params, step, loss, config
python <name>/run.py sandbox "..."   # generate code and run it (extensible)
```

The `sandbox` command is the extension point for "sandbox AI" and future tools:
it asks the model for code and runs it in a throwaway directory. It's opt-in and
easy to grow (containers, tools, agents, etc.).

---

## 3. Talk to it

### Terminal

```bash
python chat.py          # chat with the current checkpoint
python chat.py --dash   # live training dashboard (step / loss / speed / sample)
```

### Web UI

```bash
python app.py           # open http://127.0.0.1:5000
```

A small, phone-friendly page: a chat tab, and a training tab that starts/stops
training in the background and shows loss, tokens/sec, grade, and a live sample.

---

## Project layout

```
N1Bit-ARM64/
├── links.txt          # Hugging Face dataset URLs
├── requirements.txt
├── train.py           # train + export .zip
├── chat.py            # terminal chat / dashboard
├── app.py             # small Flask web UI
└── n1bit/
    ├── config.py      # all settings in one place
    ├── tokenizer.py   # byte-level tokenizer (no training step)
    ├── model.py       # the one 1-bit transformer
    ├── data.py        # smallest-first downloader -> one data.bin (resumable)
    ├── trainer.py     # training loop, resume, autosave
    └── packager.py    # builds the runnable .zip bundle
```

---

## Notes

- **PyTorch is the single backend.** CPU works out of the box; CUDA is used
  automatically if available.
- **Byte-level tokenizer** (256 bytes + PAD/BOS/EOS): no vocab to train, works for
  any language or code. Sequences are a bit longer than BPE — a fair trade for
  simplicity.
- The model is small by default (see `n1bit/config.py`); bump `EMBED_DIM`,
  `NUM_LAYERS`, `SEQ_LEN` if you have more compute.

## License

See [LICENSE](LICENSE).
