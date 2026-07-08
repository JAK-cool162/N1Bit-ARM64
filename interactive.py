import os
import sys
import json
import numpy as np
from typing import List, Dict

# Add parent directory to path to ensure correct package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from n1bit.config import (
    LINKS_FILE, CACHE_DIR, STATS_FILE,
    EMBED_DIM, NUM_LAYERS, NUM_HEADS, SEQ_LEN, get_model_paths
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
from n1bit.vulkan_engine import is_vulkan_available, get_termux_vulkan_guide
from n1bit.sandbox import ProotSandbox

# Global variables for sandbox state
sandbox_active = False
sandbox_instance = None

def list_existing_models() -> List[str]:
    """Scans cache/ directory to find all existing named models."""
    if not os.path.exists(CACHE_DIR):
        return ["default"]
        
    models = []
    for item in os.listdir(CACHE_DIR):
        item_path = os.path.join(CACHE_DIR, item)
        if os.path.isdir(item_path):
            if any(f in os.listdir(item_path) for f in ["tokenizer.json", "numpy_weights.npz", "model_checkpoint.pt"]):
                models.append(item)
    if "default" not in models:
        models.insert(0, "default")
    return models

def add_new_dataset_url():
    """Allows adding a new Hugging Face dataset URL directly into links.txt."""
    print("\n" + "="*50)
    print("          ADD A NEW HUGGING FACE DATASET URL")
    print("="*50)
    url = input("Enter Hugging Face dataset URL: ").strip()
    
    if not url:
        print("[Error] Empty URL entered.")
        return
        
    if "huggingface.co/datasets" not in url:
        print("[Warning] This URL does not look like a standard Hugging Face datasets link, but appending anyway.")
        
    existing = []
    if os.path.exists(LINKS_FILE):
        with open(LINKS_FILE, 'r', encoding='utf-8') as f:
            existing = [line.strip() for line in f if line.strip()]
            
    if url in existing:
        print("[Info] This URL is already in links.txt!")
        return
        
    with open(LINKS_FILE, 'a', encoding='utf-8') as f:
        f.write(url + "\n")
        
    print(f"[Success] Appended '{url}' successfully to '{LINKS_FILE}'!")

def load_model_dimensions(model_name: str) -> tuple:
    """Loads custom dimensions from model_config.json if it exists, otherwise uses defaults."""
    paths = get_model_paths(model_name)
    config_path = os.path.join(paths["model_dir"], "model_config.json")
    
    # Defaults
    e_dim = EMBED_DIM
    n_layers = NUM_LAYERS
    n_heads = NUM_HEADS
    s_len = SEQ_LEN
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            e_dim = cfg.get("embed_dim", EMBED_DIM)
            n_layers = cfg.get("num_layers", NUM_LAYERS)
            n_heads = cfg.get("num_heads", NUM_HEADS)
            s_len = cfg.get("seq_len", SEQ_LEN)
            print(f"[System] Loaded custom model dimensions from model_config.json (Embed: {e_dim}, Layers: {n_layers})")
        except Exception:
            pass
            
    return e_dim, n_layers, n_heads, s_len

def load_numpy_model(npz_path: str, model_name: str):
    """Helper to load .npz weights into pure NumPy BitNet model (either Transformer or RNN)."""
    data = np.load(npz_path, allow_pickle=True)
    
    model_type = "transformer"
    if "model_type" in data:
        model_type = str(data["model_type"])
        
    # Get configuration dimensions
    embed_dim, num_layers, num_heads, seq_len = load_model_dimensions(model_name)
    
    if model_type == "rnn":
        vocab_size = int(data["vocab_size"])
        
        model = NumPyBitRNNLM(vocab_size=vocab_size, embed_dim=embed_dim, seq_len=seq_len)
        model.E = data["E"]
        model.W_xh = data["W_xh"]
        model.W_hh = data["W_hh"]
        model.W_hy = data["W_hy"]
        model.b_h = data["b_h"]
        model.b_y = data["b_y"]
        return model, "rnn", seq_len
        
    else:
        weights_dict = {
            "num_heads": num_heads,
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
            
        return NumPyBitTransformerLM(weights_dict), "transformer", seq_len

def show_neuron_states(model, is_numpy: bool, model_type: str):
    """Displays neuron binary states."""
    print("\n" + "="*50)
    print("          NEURON STATE STATISTICS (1-BIT ANALYSIS)")
    print("="*50)
    
    weights = []
    if is_numpy:
        if model_type == "rnn":
            weights.extend([model.W_xh, model.W_hh, model.W_hy])
        else:
            weights.extend([model.lm_head.weight])
            for b in model.blocks:
                weights.extend([b.attn.q_proj.weight, b.attn.k_proj.weight, b.attn.v_proj.weight, b.attn.out_proj.weight])
    else:
        for name, param in model.named_parameters():
            if "weight" in name and "embedding" not in name and "ln" not in name:
                weights.append(param.detach().cpu().numpy())
                
    if not weights:
        print("Could not retrieve model weights.")
        return
        
    total_weights = 0
    total_positive = 0
    total_negative = 0
    
    for w in weights:
        signs = np.sign(w)
        total_weights += signs.size
        total_positive += np.sum(signs >= 0)
        total_negative += np.sum(signs < 0)
        
    pos_percent = (total_positive / total_weights) * 100
    neg_percent = (total_negative / total_weights) * 100
    
    print(f"Total 1-Bit Connections Checked: {total_weights:,}")
    print(f"Active Positive (+1) States:   {total_positive:,} ({pos_percent:.2f}%)")
    print(f"Active Negative (-1) States:   {total_negative:,} ({neg_percent:.2f}%)")
    print("-" * 50)
    print("Visualizing 1-bit Connection Grid (sample of 40 states):")
    flat_signs = np.concatenate([w.flatten() for w in weights])
    sample_states = np.random.choice(flat_signs, min(len(flat_signs), 40), replace=False)
    grid = "".join(["█" if s >= 0 else "░" for s in sample_states])
    print(grid)
    print("(█ = +1 neuron connection, ░ = -1 neuron connection)")
    print("="*50 + "\n")

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
    global sandbox_active, sandbox_instance
    print_banner()
    
    available_models = list_existing_models()
    print("\nAvailable Named Models inside cache directory:")
    for idx, name in enumerate(available_models, 1):
        print(f" {idx}. {name}")
        
    choice_idx = input(f"Choose model to load (1-{len(available_models)}, press Enter for default 'default'): ").strip()
    model_name = "default"
    if choice_idx.isdigit():
        idx = int(choice_idx)
        if 1 <= idx <= len(available_models):
            model_name = available_models[idx - 1]
            
    print(f"\nActive Named Model: '{model_name}'")
    paths = get_model_paths(model_name)
    
    # Load custom dimensions for the chosen model if saved
    embed_dim, num_layers, num_heads, seq_len = load_model_dimensions(model_name)
    
    tokenizer_exists = os.path.exists(paths["tokenizer"])
    pytorch_checkpoint_exists = os.path.exists(paths["checkpoint"])
    numpy_checkpoint_exists = os.path.exists(paths["numpy_weights"])
    
    print("\n[System Status]")
    print(f"- PyTorch available:      {HAS_TORCH}")
    print(f"- Vulkan GPU Compute:     {is_vulkan_available()}")
    print(f"- Tokenizer trained:      {tokenizer_exists}")
    print(f"- PyTorch Model Checkpoint:{pytorch_checkpoint_exists}")
    print(f"- NumPy Model Checkpoint:  {numpy_checkpoint_exists}")
    print(f"- Ubuntu Sandbox Mode:    {'ACTIVE' if sandbox_active else 'INACTIVE'}")
    
    if os.path.exists(paths["stats"]):
        with open(paths["stats"], 'r') as f:
            stats = json.load(f)
        print(f"- Processed Datasets:     {stats['num_datasets_processed']} / {stats['num_samples_kept']} samples kept")
        
    while True:
        print(f"\nChoose an option (Model: '{model_name}'):")
        print("1. Run Dataset Engine & Train This Model")
        print("2. Run Interactive Chat (PyTorch Inference)")
        print("3. Run Interactive Chat (Pure NumPy - Ultra-Low Power Inference)")
        print("4. Add a new Hugging Face dataset URL directly to links.txt")
        print("5. Analyze Model Weights (+1 vs -1 States)")
        print("6. Display Pre-processing Statistics")
        print("7. Toggle Ubuntu PRoot Sandbox Environment")
        print("8. Display Vulkan GPU Acceleration Info")
        print("9. Exit")
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == "1":
            print(f"\n[Starting Pre-training Pipeline for model '{model_name}']")
            limit = input("Enter max training steps (e.g. 50, or 'inf' for infinite): ").strip()
            trainer = Trainer(model_name=model_name, limit_steps=limit)
            trainer.train()
            
            tokenizer_exists = os.path.exists(paths["tokenizer"])
            pytorch_checkpoint_exists = os.path.exists(paths["checkpoint"])
            numpy_checkpoint_exists = os.path.exists(paths["numpy_weights"])
            
        elif choice == "2":
            if not HAS_TORCH:
                print("[Error] PyTorch is not available. Please use pure NumPy inference instead (Option 3).")
                continue
            if not pytorch_checkpoint_exists:
                print("[Error] PyTorch checkpoint not found. Please train the model first (Option 1).")
                continue
                
            tokenizer = SimpleBPETokenizer()
            tokenizer.load(paths["tokenizer"])
            
            print("\n[Loading PyTorch model...]")
            model = BitTransformerLM(
                vocab_size=len(tokenizer.vocab),
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                seq_len=seq_len
            )
            model.load_state_dict(torch.load(paths["checkpoint"], map_location="cpu"))
            model.eval()
            print("[Model loaded on CPU successfully.]")
            
            run_chat_loop(model, tokenizer, is_numpy=False, model_type="transformer", active_seq_len=seq_len)
            
        elif choice == "3":
            if not numpy_checkpoint_exists:
                print("[Error] NumPy checkpoint not found. Please train the model first (Option 1) to export .npz weights.")
                continue
                
            tokenizer = SimpleBPETokenizer()
            tokenizer.load(paths["tokenizer"])
            
            print("\n[Loading Pure NumPy 1-Bit Model...]")
            model, model_type, active_seq_len = load_numpy_model(paths["numpy_weights"], model_name)
            print(f"[Model ({model_type.upper()}) loaded into NumPy engine successfully. Ready for low-power ARM64 inference!]")
            
            run_chat_loop(model, tokenizer, is_numpy=True, model_type=model_type, active_seq_len=active_seq_len)
            
        elif choice == "4":
            add_new_dataset_url()
            
        elif choice == "5":
            if not numpy_checkpoint_exists and not pytorch_checkpoint_exists:
                print("[Error] No model checkpoint exists yet. Please train your model first!")
                continue
                
            if HAS_TORCH and pytorch_checkpoint_exists:
                tokenizer = SimpleBPETokenizer()
                tokenizer.load(paths["tokenizer"])
                model = BitTransformerLM(
                    vocab_size=len(tokenizer.vocab),
                    embed_dim=embed_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    seq_len=seq_len
                )
                model.load_state_dict(torch.load(paths["checkpoint"], map_location="cpu"))
                show_neuron_states(model, is_numpy=False, model_type="transformer")
            else:
                model, model_type, _ = load_numpy_model(paths["numpy_weights"], model_name)
                show_neuron_states(model, is_numpy=True, model_type=model_type)
                
        elif choice == "6":
            if os.path.exists(paths["stats"]):
                with open(paths["stats"], 'r') as f:
                    stats = json.load(f)
                print("\n" + "="*50)
                print(f"          DATASET STATISTICS: model '{model_name}'")
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
                print("[Error] No statistics file found for this model. Please run Option 1 to preprocess.")
                
        elif choice == "7":
            sandbox_active = not sandbox_active
            if sandbox_active:
                sandbox_instance = ProotSandbox()
                print(f"\n[Sandbox] Ubuntu PRoot Sandbox Environment is now ACTIVE!")
                print(f"[Sandbox] Working directory isolated to: {sandbox_instance.root_dir}/")
                if sandbox_instance.has_proot:
                    print("[Sandbox] Native 'proot' binary found on phone! Full simulation enabled.")
                else:
                    print("[Sandbox] 'proot' not found in system path. Running in secure simulated sandbox.")
            else:
                sandbox_instance = None
                print("\n[Sandbox] Ubuntu PRoot Sandbox Environment is now INACTIVE.")
                
        elif choice == "8":
            print(get_termux_vulkan_guide())
            
        elif choice == "9":
            print("\nExiting. Thank you for using N1Bit-ARM64 AI!")
            break
        else:
            print("[Error] Invalid option. Please enter 1-9.")

def run_chat_loop(model, tokenizer, is_numpy: bool, model_type: str, active_seq_len: int):
    """Interactive loop to prompt the 1-bit AI model with optional sandbox execution."""
    global sandbox_active, sandbox_instance
    print("\n" + "-"*50)
    engine_name = f"Pure NumPy 1-bit {model_type.upper()}" if is_numpy else f"PyTorch 1-bit {model_type.upper()}"
    print(f" Chatting with N1Bit-ARM64 Model ({engine_name})")
    print(f" PRoot Sandbox Mode: {'ACTIVE 🛡️' if sandbox_active else 'INACTIVE ❌'}")
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
        
        prompt_ids = prompt_ids[-active_seq_len+20:]
        prompt_ids = [tokenizer.bos_id] + prompt_ids
        
        # Thinking Log
        if is_numpy:
            context_ids = np.array([prompt_ids], dtype=np.int32)
            logits = model.forward(context_ids)
            next_logits = logits[0, -1, :].astype(np.float32)
        else:
            context_ids = torch.tensor([prompt_ids], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(context_ids)
            next_logits = logits[0, -1, :].cpu().numpy().astype(np.float32)
            
        exp_logits = np.exp(next_logits - np.max(next_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top_indices = np.argsort(probs)[::-1][:5]
        print("\nAI Thinking Logs:")
        print("--------------------------------------------------")
        for i, idx in enumerate(top_indices, 1):
            token_str = tokenizer.inverse_vocab.get(str(idx), tokenizer.inverse_vocab.get(idx, "<unk>"))
            token_str = repr(token_str)
            print(f"  {i}. {token_str:<12} | Probability: {probs[idx]*100:5.2f}%")
        print("--------------------------------------------------")
        
        # Generate Response
        output_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
        new_ids = output_ids[len(prompt_ids):]
        response = tokenizer.decode(new_ids).strip()
        
        print("\nAI Response: ")
        if not response:
            print("[N1Bit Engine is thinking...] (No response generated yet, ensure model is fully trained)")
            continue
            
        print(response)
        
        # SANDBOX INTERCEPTION
        if sandbox_active and sandbox_instance is not None:
            python_matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
            bash_matches = re.findall(r'```(?:bash|sh)\n(.*?)```', response, re.DOTALL)
            
            if python_matches:
                code_to_run = python_matches[0].strip()
                print("\n" + "="*50)
                print("🛡️ [Sandbox Guard] Python Code Block Detected in AI Output!")
                print("="*50)
                print(code_to_run)
                print("-" * 50)
                run_choice = input("Do you want to execute this Python code inside the PRoot Ubuntu Sandbox? (y/n): ").strip().lower()
                if run_choice == 'y':
                    print("\n[Sandbox] Launching script...")
                    res = sandbox_instance.execute_python_code(code_to_run)
                    print(f"\n[Sandbox Stdout]:\n{res['stdout']}")
                    if res['stderr']:
                        print(f"\n[Sandbox Stderr]:\n{res['stderr']}")
                    print(f"[Sandbox Exit Code]: {res['exit_code']}")
                    print("="*50)
                    
            elif bash_matches:
                cmd_to_run = bash_matches[0].strip()
                print("\n" + "="*50)
                print("🛡️ [Sandbox Guard] Shell Command Block Detected in AI Output!")
                print("="*50)
                print(cmd_to_run)
                print("-" * 50)
                run_choice = input("Do you want to execute this shell command inside the PRoot Ubuntu Sandbox? (y/n): ").strip().lower()
                if run_choice == 'y':
                    print("\n[Sandbox] Executing command...")
                    res = sandbox_instance.execute_command(cmd_to_run)
                    print(f"\n[Sandbox Stdout]:\n{res['stdout']}")
                    if res['stderr']:
                        print(f"\n[Sandbox Stderr]:\n{res['stderr']}")
                    print(f"[Sandbox Exit Code]: {res['exit_code']}")
                    print("="*50)
                    
        print()

if __name__ == "__main__":
    main()
