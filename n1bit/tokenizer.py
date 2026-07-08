"""Byte-level tokenizer.

There is nothing to *train* here: every byte (0-255) is its own token, and we add
three special tokens (PAD/BOS/EOS). That makes it language- and code-agnostic and
removes the whole "fit a BPE vocab" step that made the old repo feel like it was
always reloading data.
"""

from .config import PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE


class ByteTokenizer:
    vocab_size = VOCAB_SIZE
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8", errors="ignore"))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            if i < 256:
                out.append(i)
            # specials (PAD/BOS/EOS) are simply skipped
        return out.decode("utf-8", errors="ignore")
