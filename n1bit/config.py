import os
import platform

# Paths
LINKS_FILE = "links.txt"
CACHE_DIR = "cache"
PROCESSED_DATA_FILE = os.path.join(CACHE_DIR, "processed_data.jsonl")
TOKENIZER_FILE = os.path.join(CACHE_DIR, "tokenizer.json")
MODEL_CHECKPOINT = os.path.join(CACHE_DIR, "model_checkpoint.pt")
STATS_FILE = os.path.join(CACHE_DIR, "dataset_stats.json")
PROGRESS_FILE = os.path.join(CACHE_DIR, "training_progress.json")
RAW_DOWNLOADS_DIR = "downloads"  # Visible downloaded raw files folder in repo

# Create dirs if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DOWNLOADS_DIR, exist_ok=True)

# Dataset limits
MAX_SAMPLES_PER_DATASET = 2000  # Optimized for fast processing and lower CPU power
SAMPLE_QUALITY_THRESHOLD = 0.4   # Min quality score to keep sample

# Model Hyperparameters (Optimized for 16-bit float and Q1 binary quantization)
BATCH_SIZE = 16
SEQ_LEN = 64                  # Slightly shorter sequences for much faster processing on mobile
EMBED_DIM = 128               # Lower dimension to maximize training speed
NUM_LAYERS = 2
NUM_HEADS = 2
LR = 0.002
EPOCHS = 3

# Quantization and DataType Options
QUANT_TYPE = "Q1"             # "Q1" (1-bit binary: -1, +1). Extremely fast sign quantization.
USE_16BIT = True              # Use FP16 (Float16) for half-precision speed acceleration

def is_pc_environment():
    """
    Checks if the system is a PC/x86 environment or has CUDA capabilities,
    otherwise it's treated as mobile/ARM64.
    """
    machine = platform.machine().lower()
    is_x86 = "x86" in machine or "amd64" in machine or "i386" in machine or "i686" in machine
    
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass
        
    return is_x86 or has_cuda
