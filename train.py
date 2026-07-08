import os
import sys

# Add parent directory to path to ensure correct package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from n1bit.trainer import Trainer

def main():
    print("="*60)
    print("           N1Bit-ARM64 Pre-Training Runner")
    print("="*60)
    print("This script runs the complete end-to-end pipeline:")
    print("1. Reads links.txt and downloads 52 datasets using streaming.")
    print("2. Cleans, filters, de-duplicates, and quality-scores samples.")
    print("3. Generates unified dataset pre-training statistics.")
    print("4. Trains custom pure-Python BPE Tokenizer.")
    print("5. Pre-trains the custom 1-Bit AI Architecture (BitNet b1.58).")
    print("6. Exports highly optimized 1-bit weights to pure NumPy engine.")
    print("="*60)
    
    # Check if a steps limit is provided
    limit_steps = None
    if len(sys.argv) > 1:
        try:
            limit_steps = int(sys.argv[1])
            print(f"Limiting training to {limit_steps} steps.")
        except ValueError:
            pass
            
    trainer = Trainer(limit_steps=limit_steps)
    trainer.train()

if __name__ == "__main__":
    main()
