import os
import platform

# Paths
LINKS_FILE = "links.txt"
CACHE_DIR = "cache"
PROCESSED_DATA_FILE = os.path.join(CACHE_DIR, "processed_data.jsonl")
TOKENIZER_FILE = os.path.join(CACHE_DIR, "tokenizer.json")
MODEL_CHECKPOINT = os.path.join(CACHE_DIR, "model_checkpoint.pt")
STATS_FILE = os.path.join(CACHE_DIR, "dataset_stats.json")

# Create CACHE_DIR if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset limits
MAX_SAMPLES_PER_DATASET = 5000  # Stream limit to avoid infinite loops and low RAM usage
SAMPLE_QUALITY_THRESHOLD = 0.4   # Min quality score to keep sample

# Model Hyperparameters (optimized for ultra-low power and mobile compatibility)
BATCH_SIZE = 32
SEQ_LEN = 128
EMBED_DIM = 256
NUM_LAYERS = 3
NUM_HEADS = 4
LR = 0.001
EPOCHS = 1

def is_pc_environment():
    """
    Checks if the system is a PC/x86 environment or has CUDA capabilities,
    otherwise it's treated as mobile/ARM64.
    """
    machine = platform.machine().lower()
    is_x86 = "x86" in machine or "amd64" in machine or "i386" in machine or "i686" in machine
    
    # Check for PyTorch & CUDA
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass
        
    return is_x86 or has_cuda
