import os
import sys

# Add parent directory to path to ensure correct package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from n1bit.trainer import Trainer
from n1bit.dataset import DatasetEngine
from n1bit.config import LINKS_FILE, get_model_paths

def show_dataset_selector() -> list:
    """Displays a numbered list of all 52 datasets and allows the user to select subset."""
    print("\n" + "="*60)
    print("          CHOOSE DATASETS FOR YOUR NAMED MODEL")
    print("="*60)
    
    # Read links
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
    
    choice = input("\nYour selection: ").strip()
    if not choice or choice.lower() == "all":
        print("Selected ALL datasets.")
        return None
        
    selected_repos = []
    try:
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

def main():
    print("="*60)
    print("           N1Bit-ARM64 Pre-Training Runner")
    print("="*60)
    
    # Defaults
    model_name = "default"
    limit_steps = None
    selected_repos = None
    
    args = sys.argv[1:]
    
    # Parse arguments
    # 1. Custom model name check: python train.py name coder [steps]
    if len(args) >= 2 and args[0].lower() == "name":
        model_name = args[1]
        print(f"Initializing named model: '{model_name}'")
        
        # Interactive dataset selection!
        selected_repos = show_dataset_selector()
        
        # Check if third argument is steps (e.g., inf or 50)
        if len(args) >= 3:
            limit_steps = args[2]
            
    # 2. Standard step check: python train.py [steps] (like python train.py 50 or python train.py inf)
    elif len(args) >= 1:
        limit_steps = args[0]
        
    if limit_steps == "inf":
        print(f"Training of model '{model_name}' will run INFINITELY (Epochs loop infinitely).")
    elif limit_steps:
        print(f"Training step limit: {limit_steps} steps.")
    else:
        print("Standard training mode (default epoch limits).")
        
    # Start Trainer
    trainer = Trainer(model_name=model_name, limit_steps=limit_steps)
    trainer.train(selected_repos=selected_repos)

if __name__ == "__main__":
    main()
