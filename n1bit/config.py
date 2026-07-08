"""All settings in one small place. Tweak these; there is nothing hidden elsewhere."""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LINKS_FILE = "links.txt"          # list of Hugging Face dataset URLs
CACHE_DIR = "cache"               # everything the tool generates lives here
DATA_BIN = os.path.join(CACHE_DIR, "data.bin")        # the ONE training file
MANIFEST = os.path.join(CACHE_DIR, "manifest.json")   # what we've downloaded
CHECKPOINT = os.path.join(CACHE_DIR, "model.pt")      # resumable checkpoint
BUILDS_DIR = "builds"             # finished .zip bundles land here

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BUILDS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tokenizer  (byte-level: every byte 0..255 is a token, plus a few specials)
# ---------------------------------------------------------------------------
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259

# ---------------------------------------------------------------------------
# Model  (small on purpose — this runs on a phone)
# ---------------------------------------------------------------------------
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
SEQ_LEN = 128

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
LR = 3e-4
SAVE_EVERY = 25          # steps between autosaves
LOG_EVERY = 5            # steps between dashboard updates
DEFAULT_STEPS = 2000     # `python train.py` with no args runs this many steps
