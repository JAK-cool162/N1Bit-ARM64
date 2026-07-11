import os
import sys
import json
import argparse

# Add parent directory to path to ensure correct package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from n1bit.trainer import Trainer
from n1bit.dataset import DatasetEngine
from n1bit.config import (
    LINKS_FILE, MODEL_PROFILES, calculate_parameter_count, get_model_paths,
    EMBED_DIM, NUM_LAYERS, NUM_HEADS, SEQ_LEN, LR
)

def show_dataset_selector() -> list:
    """Displays a numbered list of all 52 datasets and allows the user to select a subset."""
    print("\n" + "="*60)
    print("          CHOOSE DATASETS FOR YOUR NAMED MODEL")
    print("="*60)
    
    temp_engine = DatasetEngine("cache/temp.jsonl", "cache/temp.json")
    urls = temp_engine.read_links()
    
    if not urls:
        print("No datasets found in links.txt. Training on offline fallback.")
        return None
        
    repos = [temp_engine.parse_repo_id(url) for url in urls]
    
    for idx, repo in enumerate(repos, 1):
        print(f"[{idx:2d}] {repo}")
        
    print("\nOptions:")
    print("- Type numbers separated by commas (e.g., '1, 3, 10, 15') to select specific datasets.")
    print("- Type 'all' or press Enter to train on ALL datasets.")
    
    # Non-interactive terminal safety
    if not sys.stdin.isatty():
        print("Non-interactive terminal detected, selecting ALL datasets by default.")
        return None
        
    try:
        choice = input("\nYour selection: ").strip()
        if not choice or choice.lower() == "all":
            print("Selected ALL datasets.")
            return None
            
        selected_repos = []
        indices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
        for idx in indices:
            if 1 <= idx <= len(repos):
                selected_repos.append(repos[idx - 1])
        if selected_repos:
            print(f"\nSelected {len(selected_repos)} datasets for training: {selected_repos}")
            return selected_repos
    except Exception:
        pass
        
    print("Invalid choice, defaulting to ALL datasets.")
    return None

def show_profile_selector() -> dict:
    """Displays hardware-optimised profiles and returns selected config details."""
    print("\n" + "="*60)
    print("      SELECT YOUR TARGET HARDWARE CHIPSET PROFILE")
    print("="*60)
    print("Cramming a massive model into a mobile CPU will freeze your phone.")
    print("Select a lightweight, low-power preset balanced for your device:")
    print("-" * 60)
    
    for key, p in MODEL_PROFILES.items():
        print(f"[{key}] {p['name'].upper()} PROFILE ({p['target_params']} parameters)")
        print(f"    - Dimensions: Embed: {p['embed_dim']} | Layers: {p['num_layers']} | Heads: {p['num_heads']} | Context: {p['seq_len']}")
        print(f"    - Target Chips: {p['description']}")
        print(f"    - Recommendation: {p['recommendation']}\n")
        
    # Non-interactive safety
    if not sys.stdin.isatty():
        print("Non-interactive terminal detected, selecting MIDRANGE profile by default.")
        return MODEL_PROFILES["2"]
        
    try:
        choice = input("Select profile (1-4, press Enter for standard '2' Midrange): ").strip()
        if not choice:
            choice = "2"
    except Exception:
        choice = "2"
        
    profile = MODEL_PROFILES.get(choice, MODEL_PROFILES["2"])
    print(f"\n[Selected Profile]: {profile['name'].upper()}")
    return profile

def main():
    print("="*60)
    print("           N1Bit-ARM64 Pre-Training Runner")
    print("="*60)
    
    # 1. Setup argparse for robust parsing (Colab & CLI friendly)
    parser = argparse.ArgumentParser(description="N1Bit-ARM64 Pre-Training Runner")
    parser.add_argument("steps_pos", nargs="?", default=None, help="Steps (positional argument fallback)")
    parser.add_argument("--steps", "-s", default=None, help="Number of training steps or 'inf'")
    parser.add_argument("--name", "-n", default="default", help="Name of the model to train")
    parser.add_argument("--profile", "-p", default=None, help="Target profile index (1-4)")
    
    args, unknown = parser.parse_known_args()
    
    # Defaults
    model_name = args.name
    limit_steps = args.steps if args.steps is not None else args.steps_pos
    selected_repos = None
    profile_id = args.profile
    
    # Selected dimensions defaults (Midrange Profile)
    embed_dim = EMBED_DIM
    num_layers = NUM_LAYERS
    num_heads = NUM_HEADS
    seq_len = SEQ_LEN
    
    # 2. Handled word-based positional fallbacks: e.g. python train.py name coder 500000
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "name":
        model_name = sys.argv[2]
        print(f"Initializing named model: '{model_name}'")
        
        # Hardware Profile Selection
        if profile_id is None:
            profile = show_profile_selector()
        else:
            profile = MODEL_PROFILES.get(profile_id, MODEL_PROFILES["2"])
            
        embed_dim = profile["embed_dim"]
        num_layers = profile["num_layers"]
        num_heads = profile["num_heads"]
        seq_len = profile["seq_len"]
        
        # Dataset selection
        selected_repos = show_dataset_selector()
        
        if len(sys.argv) >= 4:
            limit_steps = sys.argv[3]
            
    # Resolve step count variables
    if limit_steps is not None:
        # Strip potential --steps if parsed improperly
        limit_steps = str(limit_steps).replace("--steps", "").strip()
        if limit_steps == "inf":
            pass
        elif limit_steps.isdigit():
            limit_steps = int(limit_steps)
        else:
            # Handle empty or invalid formats
            limit_steps = None

    paths = get_model_paths(model_name)
    config_path = os.path.join(paths["model_dir"], "model_config.json")
    
    # Estimate total parameters
    estimated_params = calculate_parameter_count(4000, embed_dim, num_layers, seq_len)
    print(f"\n[Model Architecture Details]: '{model_name}'")
    print(f"  - Embedding Dimension:  {embed_dim}")
    print(f"  - Attention Heads:      {num_heads}")
    print(f"  - Transformer Layers:   {num_layers}")
    print(f"  - Max Sequence Length:  {seq_len}")
    print(f"  - Estimated Parameter Count: {estimated_params:,} parameters")
    print(f"  - Memory footprint: ~{estimated_params * 2 / 1024:.2f} KB (at FP16 half-precision)")
    print("-" * 60)
    
    # Write config to model-specific folder
    model_config = {
        "model_name": model_name,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "estimated_params": estimated_params
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved model architecture parameters to {config_path}")
    
    if limit_steps == "inf":
        print(f"Training of model '{model_name}' will run INFINITELY (Epochs loop infinitely).")
    elif limit_steps:
        print(f"Training step limit: {limit_steps} steps.")
    else:
        print("Standard training mode (default epoch limits).")
        
    # Start Trainer
    trainer = Trainer(model_name=model_name, limit_steps=limit_steps)
    
    # Inject dimensions dynamically
    import n1bit.trainer as t_mod
    t_mod.EMBED_DIM = embed_dim
    t_mod.NUM_LAYERS = num_layers
    t_mod.NUM_HEADS = num_heads
    t_mod.SEQ_LEN = seq_len
    
    trainer.train(selected_repos=selected_repos)

if __name__ == "__main__":
    main()
