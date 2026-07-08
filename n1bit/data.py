"""Dataset builder.

Old behaviour (the thing you didn't like): every run re-streamed and re-filtered all
52 datasets, tried to parse images / minecraft / chemistry tables, and only at the end
turned it into a .bin.

New behaviour:
  1. Read links.txt.
  2. Ask the Hugging Face API which text file in each repo is best, and how big it is.
  3. Sort datasets SMALLEST -> LARGEST and download in that order, so usable data
     arrives fast and you can stop whenever you want.
  4. Stream each file, pull out text, and append the tokens straight into ONE file:
     cache/data.bin  (uint16 array of byte-level token ids).
  5. Record every finished repo in cache/manifest.json so the next run SKIPS them
     instead of downloading again.

Run it directly:
    python -m n1bit.data              # download all, smallest first
    python -m n1bit.data --max 5      # only the 5 smallest
    python -m n1bit.data --reset      # wipe data.bin + manifest and start over
"""

import os
import re
import io
import csv
import json
import time
import array
import argparse
import urllib.request
import urllib.error

from .config import LINKS_FILE, DATA_BIN, MANIFEST, EOS_ID
from .tokenizer import ByteTokenizer

HF_API = "https://huggingface.co/api/datasets/{repo}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
# .parquet is the most common HF format; we read it if pyarrow/pandas is available.
TEXT_EXT = (".jsonl", ".json", ".txt", ".csv", ".parquet")
# fields we treat as training text, in rough priority order
TEXT_KEYS = ("text", "content", "output", "response", "completion", "answer",
             "instruction", "prompt", "question", "input", "message")
UA = {"User-Agent": "n1bit-arm64/2.0"}


def _get(url, timeout=30, retries=3):
    """Open a URL with a few retries for flaky connections. Returns the response."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - network is unpredictable
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def read_links() -> list[str]:
    if not os.path.exists(LINKS_FILE):
        return []
    out = []
    with open(LINKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def repo_id(url: str) -> str:
    m = re.search(r"huggingface\.co/datasets/([^/\s]+/[^/\s]+)", url)
    return m.group(1) if m else url.rstrip("/").split("datasets/")[-1]


def pick_file(repo: str):
    """Return (path, size_bytes) of the smallest usable text file.

    Returns None if the repo genuinely has no text/parquet file, or the string
    "unreachable" if the Hugging Face API couldn't be contacted (so the caller
    can tell a real skip apart from a network hiccup).
    """
    try:
        data = json.loads(_get(HF_API.format(repo=repo), timeout=15).read())
    except Exception:
        return "unreachable"
    best = None
    for s in data.get("siblings", []):
        name = s.get("rfilename", "")
        if not name.lower().endswith(TEXT_EXT):
            continue
        size = s.get("size")
        if size is None:  # size not in listing; fetch via HEAD
            size = _head_size(HF_RESOLVE.format(repo=repo, path=name))
        if size is None:
            size = 10 ** 12  # unknown -> treat as huge so it sorts last
        if best is None or size < best[1]:
            best = (name, size)
    return best


def _head_size(url):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def clean(text) -> str | None:
    if not isinstance(text, str):
        return None
    t = "".join(c for c in text if c == "\n" or 32 <= ord(c) < 0x110000).strip()
    return t if len(t) >= 20 else None


def extract(obj):
    """Yield clean text strings from one parsed record."""
    if isinstance(obj, str):
        c = clean(obj)
        if c:
            yield c
    elif isinstance(obj, dict):
        # chat-style
        msgs = obj.get("messages") or obj.get("conversations")
        if isinstance(msgs, list):
            parts = [clean(m.get("content") or m.get("value")) for m in msgs
                     if isinstance(m, dict)]
            joined = "\n".join(p for p in parts if p)
            if joined:
                yield joined
                return
        # prioritized fields, else any string value
        emitted = False
        for k in TEXT_KEYS:
            if k in obj:
                c = clean(obj[k])
                if c:
                    yield c
                    emitted = True
        if not emitted:
            for v in obj.values():
                c = clean(v)
                if c:
                    yield c


def _iter_records(path, raw):
    """Yield parsed records from a downloaded file's bytes."""
    text = raw.decode("utf-8", errors="ignore")
    low = path.lower()
    if low.endswith(".jsonl"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
    elif low.endswith(".json"):
        try:
            data = json.loads(text)
        except Exception:
            return
        yield from (data if isinstance(data, list) else [data])
    elif low.endswith(".csv"):
        for row in csv.DictReader(io.StringIO(text)):
            yield row
    else:  # .txt
        for para in text.split("\n\n"):
            yield para


def _iter_parquet(raw: bytes):
    """Yield dict records from parquet bytes, if pyarrow or pandas is available."""
    try:
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(io.BytesIO(raw))
        for batch in table.to_batches():
            for row in batch.to_pylist():
                yield row
        return
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(io.BytesIO(raw))
        for _, row in df.iterrows():
            yield row.to_dict()
    except Exception:
        print("[data]     parquet support needs pyarrow or pandas; skipping")


def build(max_datasets: int | None = None, per_dataset_bytes: int = 20_000_000,
          reset: bool = False):
    tok = ByteTokenizer()

    if reset:
        for p in (DATA_BIN, MANIFEST):
            if os.path.exists(p):
                os.remove(p)

    done = set()
    if os.path.exists(MANIFEST):
        done = set(json.load(open(MANIFEST)).get("datasets", []))

    links = read_links()
    print(f"[data] {len(links)} datasets in {LINKS_FILE}, {len(done)} already done")

    # Discover sizes, skipping finished ones, then sort smallest first.
    todo = []
    unreachable = 0
    for url in links:
        repo = repo_id(url)
        if repo in done:
            continue
        info = pick_file(repo)
        if info == "unreachable":
            unreachable += 1
            print(f"[data]   ????  {repo} (network unreachable, will retry next run)")
        elif info:
            todo.append((info[1], repo, info[0]))
            print(f"[data]   found {repo} -> {info[0]} ({info[1] / 1e6:.1f} MB)")
        else:
            print(f"[data]   skip  {repo} (no usable text/parquet file)")
    if unreachable:
        print(f"[data] {unreachable} datasets unreachable (network) — rerun later to add them")
    todo.sort(key=lambda t: t[0])  # smallest -> largest

    if max_datasets:
        todo = todo[:max_datasets]
    print(f"[data] downloading {len(todo)} datasets, smallest first")

    total_tokens = _bin_len()
    for size, repo, path in todo:
        print(f"[data] >>> {repo} ({size / 1e6:.1f} MB)")
        try:
            raw = _get(HF_RESOLVE.format(repo=repo, path=path), timeout=120).read(
                per_dataset_bytes
            )
        except Exception as e:
            print(f"[data]     download failed: {e}")
            continue

        buf = array.array("H")  # uint16
        kept = 0
        records = _iter_parquet(raw) if path.lower().endswith(".parquet") \
            else _iter_records(path, raw)
        for rec in records:
            for txt in extract(rec):
                buf.extend(tok.encode(txt))
                buf.append(EOS_ID)
                kept += 1
        _append_bin(buf)
        total_tokens += len(buf)
        done.add(repo)
        _save_manifest(done)
        print(f"[data]     kept {kept:,} samples, +{len(buf):,} tokens "
              f"(total {total_tokens:,})")

    print(f"[data] done. data.bin = {total_tokens:,} tokens "
          f"({os.path.getsize(DATA_BIN) / 1e6:.1f} MB)" if total_tokens else
          "[data] nothing downloaded")


def _bin_len() -> int:
    return os.path.getsize(DATA_BIN) // 2 if os.path.exists(DATA_BIN) else 0


def _append_bin(buf: "array.array"):
    with open(DATA_BIN, "ab") as f:
        buf.tofile(f)


def _save_manifest(done: set):
    json.dump({"datasets": sorted(done)}, open(MANIFEST, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description="Build cache/data.bin, smallest first")
    ap.add_argument("--max", type=int, default=None,
                    help="only download the N smallest datasets")
    ap.add_argument("--reset", action="store_true",
                    help="delete data.bin + manifest and start fresh")
    args = ap.parse_args()
    build(max_datasets=args.max, reset=args.reset)


if __name__ == "__main__":
    main()
