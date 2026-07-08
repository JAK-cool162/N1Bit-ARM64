import os
import sys
import numpy as np

# Add parent directory to path to ensure correct package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from n1bit.config import (
    MODEL_CHECKPOINT, TOKENIZER_FILE, STATS_FILE,
    EMBED_DIM, NUM_LAYERS, NUM_HEADS, SEQ_LEN
)
from n1bit.tokenizer import SimpleBPETokenizer
from n1bit.trainer import Trainer

# Try to import torch-dependent components
try:
    import torch
    from n1bit.model import BitTransformerLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from n1bit.model import NumPyBitTransformerLM, NumPyBitRNNLM

def load_numpy_model(npz_path: str):
    """
    Helper to load .npz weights into pure NumPy BitNet model (either Transformer or RNN).
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Read model type
    model_type = "transformer"
    if "model_type" in data:
        model_type = str(data["model_type"])
        
    if model_type == "rnn":
        vocab_size = int(data["vocab_size"])
        embed_dim = int(data["embed_dim"])
        seq_len = int(data["seq_len"])
        
        # Instantiate RNN model
        model = NumPyBitRNNLM(vocab_size=vocab_size, embed_dim=embed_dim, seq_len=seq_len)
        model.E = data["E"]
        model.W_xh = data["W_xh"]
        model.W_hh = data["W_hh"]
        model.W_hy = data["W_hy"]
        model.b_h = data["b_h"]
        model.b_y = data["b_y"]
        return model, "rnn"
        
    else:
        weights_dict = {
            "num_heads": int(data["num_heads"]),
            "token_embedding": data["token_embedding"],
            "position_embedding": data["position_embedding"],
            "ln_f_w": data["ln_f_w"],
            "ln_f_b": data["ln_f_b"],
            "lm_head_w": data["lm_head_w"],
            "blocks": []
        }
        
        block_keys = [k for k in data.keys() if k.startswith("block_")]
        num_blocks = len(set(int(k.split("_")[1]) for k in block_keys))
        
        for i in range(num_blocks):
            block_data = {
                "ln1_w": data[f"block_{i}_ln1_w"],
                "ln1_b": data[f"block_{i}_ln1_b"],
                "ln2_w": data[f"block_{i}_ln2_w"],
                "ln2_b": data[f"block_{i}_ln2_b"],
                "q_proj_w": data[f"block_{i}_q_proj_w"],
                "k_proj_w": data[f"block_{i}_k_proj_w"],
                "v_proj_w": data[f"block_{i}_v_proj_w"],
                "out_proj_w": data[f"block_{i}_out_proj_w"],
                "gate_proj_w": data[f"block_{i}_gate_proj_w"],
                "down_proj_w": data[f"block_{i}_down_proj_w"]
            }
            weights_dict["blocks"].append(block_data)
            
        return NumPyBitTransformerLM(weights_dict), "transformer"

def print_banner():
    print("="*60)
    print("     _   _  _ ____  _ _   _              _  _    ___    ")
    print("    | \\ | |/ |  _ \\(_) |_| |    ___ ___ | || |  / _ \\   ")
    print("    |  \\| |  | |_) | | __| |   / __/ _ \\| || |_| | | |  ")
    print("    | |\\  |  |  _ <| | |_|_|  | (_| (_) |__   _| |_| |  ")
    print("    |_| \\_|_|_|_| \\_\\_|\\__(_)   \\___\\___/   |_|  \\___/  ")
    print("                                                        ")
    print("    1-Bit AI Engine (BitNet b1.58) optimized for ARM64/Mobile")
    print("="*60)

def main():
    print_banner()
    
    tokenizer_exists = os.path.exists(TOKENIZER_FILE)
    pytorch_checkpoint_exists = os.path.exists(MODEL_CHECKPOINT)
    
    numpy_checkpoint_path = os.path.join(os.path.dirname(MODEL_CHECKPOINT), "numpy_weights.npz")
    numpy_checkpoint_exists = os.path.exists(numpy_checkpoint_path)
    
    print("\n[System Status]")
    print(f"- PyTorch available:      {HAS_TORCH}")
    print(f"- Tokenizer trained:      {tokenizer_exists}")
    print(f"- PyTorch Model Checkpoint:{pytorch_checkpoint_exists}")
    print(f"- NumPy Model Checkpoint:  {numpy_checkpoint_exists}")
    
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
        print(f"- Processed Datasets:     {stats['num_datasets_processed']} / {stats['num_samples_kept']} samples kept")
        
    while True:
        print("\nChoose an option:")
        print("1. Run Dataset Engine & Train 1-Bit AI model")
        print("2. Run Interactive Chat (PyTorch Inference)")
        print("3. Run Interactive Chat (Pure NumPy - Ultra-Low Power Inference)")
        print("4. Display Pre-processing Statistics")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n[Starting Pre-training Pipeline]")
            limit = input("Enter max training steps (press Enter for unlimited): ").strip()
            limit_steps = int(limit) if limit.isdigit() else None
            
            trainer = Trainer(limit_steps=limit_steps)
            trainer.train()
            
            tokenizer_exists = os.path.exists(TOKENIZER_FILE)
            pytorch_checkpoint_exists = os.path.exists(MODEL_CHECKPOINT)
            numpy_checkpoint_exists = os.path.exists(numpy_checkpoint_path)
            
        elif choice == "2":
            if not HAS_TORCH:
                print("[Error] PyTorch is not available. Please use pure NumPy inference instead (Option 3).")
                continue
            if not pytorch_checkpoint_exists:
                print("[Error] PyTorch checkpoint not found. Please train the model first (Option 1).")
                continue
                
            tokenizer = SimpleBPETokenizer()
            tokenizer.load(TOKENIZER_FILE)
            
            print("\n[Loading PyTorch model...]")
            model = BitTransformerLM(
                vocab_size=len(tokenizer.vocab),
                embed_dim=EMBED_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                seq_len=SEQ_LEN
            )
            model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location="cpu"))
            model.eval()
            print("[Model loaded on CPU successfully.]")
            
            run_chat_loop(model, tokenizer, is_numpy=False, model_type="transformer")
            
        elif choice == "3":
            if not numpy_checkpoint_exists:
                print("[Error] NumPy checkpoint not found. Please train the model first (Option 1) to export .npz weights.")
                continue
                
            tokenizer = SimpleBPETokenizer()
            tokenizer.load(TOKENIZER_FILE)
            
            print("\n[Loading Pure NumPy 1-Bit Model...]")
            model, model_type = load_numpy_model(numpy_checkpoint_path)
            print(f"[Model ({model_type.upper()}) loaded into NumPy engine successfully. Ready for low-power ARM64 inference!]")
            
            run_chat_loop(model, tokenizer, is_numpy=True, model_type=model_type)
            
        elif choice == "4":
            if os.path.exists(STATS_FILE):
                with open(STATS_FILE, 'r') as f:
                    stats = json.load(f)
                print("\n" + "="*50)
                print("          DATASET PROCESSING STATISTICS")
                print("="*50)
                print(f"Datasets Processed:        {stats['num_datasets_processed']}")
                print(f"Files/Splits Processed:    {stats['num_files_processed']}")
                print(f"Samples Kept (Passed QC):  {stats['num_samples_kept']}")
                print(f"Samples Discarded:         {stats['num_samples_discarded']}")
                print(f"Duplicate Rate:            {stats['duplicate_rate']}%")
                print(f"Language Distribution:     {stats['language_distribution']}")
                print(f"Size Before Cleaning:      {stats['size_before_bytes'] / (1024*1024):.2f} MB")
                print(f"Size After Cleaning:       {stats['size_after_bytes'] / (1024*1024):.2f} MB")
                print(f"Estimated Token Count:     {stats['estimated_token_count']}")
                print("="*50 + "\n")
            else:
                print("[Error] No statistics file found. Please run Option 1 to preprocess the datasets.")
                
        elif choice == "5":
            print("\nExiting. Thank you for using N1Bit-ARM64 AI!")
            break
        else:
            print("[Error] Invalid option. Please enter 1-5.")

def run_chat_loop(model, tokenizer, is_numpy: bool, model_type: str):
    """Interactive loop to prompt the 1-bit AI model."""
    print("\n" + "-"*50)
    engine_name = f"Pure NumPy 1-bit {model_type.upper()}" if is_numpy else f"PyTorch 1-bit {model_type.upper()}"
    print(f" Chatting with N1Bit-ARM64 Model ({engine_name})")
    print(" Type 'exit' or 'quit' to return to main menu.")
    print("-"*50)
    
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            break
            
        if not prompt:
            continue
            
        formatted_prompt = f"<instruction>: {prompt}\n<response>:"
        prompt_ids = tokenizer.encode(formatted_prompt)
        
        # Limit prompt ids to fit the context window
        prompt_ids = prompt_ids[-SEQ_LEN+20:]
        prompt_ids = [tokenizer.bos_id] + prompt_ids
        
        print("\nAI: ", end="", flush=True)
        
        # Generation
        output_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
            
        num_prompt_tokens = len(prompt_ids)
        new_ids = output_ids[num_prompt_tokens:]
        
        response = tokenizer.decode(new_ids)
        if not response.strip():
            print("[N1Bit Engine is thinking...] (No response generated yet, ensure model is fully trained)")
        else:
            print(response.strip())
        print()

if __name__ == "__main__":
    main()
