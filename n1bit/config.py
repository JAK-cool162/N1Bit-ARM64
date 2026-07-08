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

# Model Hyperparameters Defaults (Midrange Profile)
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

# =====================================================================
# HARDWARE CHIPSET PROFILE DEFINITIONS
# =====================================================================

MODEL_PROFILES = {
    "1": {
        "name": "budget",
        "embed_dim": 64,
        "num_layers": 1,
        "num_heads": 1,
        "seq_len": 32,
        "target_params": "~10k - 50k",
        "description": "Ultra-Low Power: For budget/older chips (Snapdragon 400/600, MediaTek Helio, 2GB-3GB RAM phones)",
        "recommendation": "Runs instantly at 0% battery drainage. Ideal for legacy/wearable hardware."
    },
    "2": {
        "name": "midrange",
        "embed_dim": 128,
        "num_layers": 2,
        "num_heads": 2,
        "seq_len": 64,
        "target_params": "~100k - 500k",
        "description": "Balanced Power: For mid-range chips (Snapdragon 700 series, Dimensity 700/800, Apple A12/A13)",
        "recommendation": "Exceptional balance of speed, capability, and battery efficiency."
    },
    "3": {
        "name": "flagship",
        "embed_dim": 256,
        "num_layers": 3,
        "num_heads": 4,
        "seq_len": 128,
        "target_params": "~1M - 5M",
        "description": "Flagship Power: For flagship chips (Snapdragon 8 Gen 1/2/3, Dimensity 9000+, Apple A15+ / A16)",
        "recommendation": "Maximum capability on modern high-end phones. Incredible text context length."
    },
    "4": {
        "name": "desktop",
        "embed_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "seq_len": 256,
        "target_params": "10M+",
        "description": "Unlimited Power: Optimized for Desktop PC and CUDA/Vulkan GPU acceleration",
        "recommendation": "Best for PCs with dedicated graphics to train very capable larger models."
    }
}

def calculate_parameter_count(vocab_size: int, embed_dim: int, num_layers: int, seq_len: int) -> int:
    """Estimates total parameters to prevent memory crashes."""
    token_emb = vocab_size * embed_dim
    pos_emb = seq_len * embed_dim
    
    params_per_block = 12 * (embed_dim * embed_dim) + (4 * embed_dim)
    total_blocks_params = num_layers * params_per_block
    
    final_ln = 2 * embed_dim
    lm_head = embed_dim * vocab_size
    
    return token_emb + pos_emb + total_blocks_params + final_ln + lm_head

def get_model_paths(model_name: str = "default") -> dict:
    """
    Dynamically returns isolated cache and file paths for a named model.
    Saves compiled training corpora as high-performance pre-tokenized .bin files
    (uint16 integers) to achieve the absolute fastest load times and bypass tokenizer overhead.
    """
    model_dir = os.path.join(CACHE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    return {
        "model_dir": model_dir,
        "processed_data": os.path.join(model_dir, "processed_data.bin"),  # Pre-tokenized binary format
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
