import os
import platform

# Base Paths
LINKS_FILE = "links.txt"
CACHE_DIR = "cache"
RAW_DOWNLOADS_DIR = "downloads"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DOWNLOADS_DIR, exist_ok=True)

# Dataset limits
MAX_SAMPLES_PER_DATASET = 2000
SAMPLE_QUALITY_THRESHOLD = 0.4

# Model Hyperparameters
BATCH_SIZE = 16
SEQ_LEN = 64
EMBED_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 2
LR = 0.002
EPOCHS = 3

# Quantization and DataType Options
QUANT_TYPE = "Q1"             # "Q1" binary quantization (-1, +1)
USE_16BIT = True              # Use FP16 (Float16) half-precision

def get_model_paths(model_name: str = "default") -> dict:
    """
    Dynamically returns isolated cache and checkpoint file paths
    for a specific named model.
    """
    model_dir = os.path.join(CACHE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    return {
        "model_dir": model_dir,
        "processed_data": os.path.join(model_dir, "processed_data.jsonl"),
        "tokenizer": os.path.join(model_dir, "tokenizer.json"),
        "checkpoint": os.path.join(model_dir, "model_checkpoint.pt"),
        "numpy_weights": os.path.join(model_dir, "numpy_weights.npz"),
        "stats": os.path.join(model_dir, "dataset_stats.json"),
        "progress": os.path.join(model_dir, "training_progress.json")
    }

def is_pc_environment():
    """Checks if current platform has x86 hardware or CUDA GPU acceleration."""
    machine = platform.machine().lower()
    is_x86 = "x86" in machine or "amd64" in machine or "i386" in machine or "i686" in machine
    
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass
        
    return is_x86 or has_cuda
