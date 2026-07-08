import os
import time
import numpy as np
from typing import List

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from .model import BitTransformerLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import (
    BATCH_SIZE, SEQ_LEN, EMBED_DIM, NUM_LAYERS, NUM_HEADS, LR, EPOCHS,
    TOKENIZER_FILE, MODEL_CHECKPOINT, PROCESSED_DATA_FILE
)
from .utils import optimize_environment
from .tokenizer import SimpleBPETokenizer
from .dataset import DatasetEngine
from .model import NumPyBitRNNLM

class Trainer:
    """
    Main Trainer for the 1-Bit AI model.
    Handles Tokenizer training, dynamic SFT/Pre-training stream packing.
    Enables automatic zero-dependency NumPy 1-bit RNN training if PyTorch is absent,
    and high-performance PyTorch BitNet (BitTransformerLM) if PyTorch is available.
    """
    def __init__(self, limit_steps: int = None):
        self.limit_steps = limit_steps
        self.engine = DatasetEngine()
        self.tokenizer = SimpleBPETokenizer(vocab_size=4000)
        
        optimize_environment()
        
    def prepare_tokenizer(self):
        """
        Ensures the tokenizer is trained and ready.
        If tokenizer does not exist, trains it on a portion of the processed dataset.
        """
        if os.path.exists(TOKENIZER_FILE):
            print(f"[Trainer] Loading existing tokenizer from {TOKENIZER_FILE}")
            self.tokenizer.load(TOKENIZER_FILE)
            return

        print("[Trainer] Tokenizer cache not found. Training tokenizer on clean dataset sample...")
        texts = []
        count = 0
        for sample in self.engine.stream_processed_samples():
            texts.append(sample["text"])
            count += 1
            if count >= 1000:
                break
                
        if not texts:
            texts = ["Hello, this is a fallback training sentence for our 1-bit ARM64 model."]
            
        self.tokenizer.train_from_texts(texts)
        self.tokenizer.save(TOKENIZER_FILE)
        print(f"[Trainer] Tokenizer trained with vocab size {len(self.tokenizer.vocab)} and saved to {TOKENIZER_FILE}")

    def get_token_chunk_stream(self) -> List[int]:
        """
        Generator that streams tokens from processed text files,
        BOS/EOS packs them, and yields segments of exactly SEQ_LEN + 1.
        """
        buffer = []
        for sample in self.engine.stream_processed_samples():
            text = sample["text"]
            tokens = [self.tokenizer.bos_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_id]
            buffer.extend(tokens)
            
            while len(buffer) >= SEQ_LEN + 1:
                chunk = buffer[:SEQ_LEN + 1]
                buffer = buffer[SEQ_LEN + 1:]
                yield chunk

    def get_batch_generator(self, chunk_generator, batch_size: int):
        """
        Groups sequence chunks into batches.
        """
        batch = []
        for chunk in chunk_generator:
            batch.append(chunk)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            while len(batch) < batch_size:
                batch.append([self.tokenizer.pad_id] * (SEQ_LEN + 1))
            yield batch

    def train(self):
        """
        Trains the 1-bit Model on the streamed dataset.
        Runs PyTorch BitNet if PyTorch is installed, otherwise falls back to
        a pure NumPy Recurrent 1-bit model (BPTT + STE + Adam) for zero-dependency mobile environments.
        """
        # 1. Run Dataset Preprocessing and stats
        self.engine.process_all_datasets()
        
        # 2. Prepare Tokenizer
        self.prepare_tokenizer()
        
        vocab_size = len(self.tokenizer.vocab)
        
        if HAS_TORCH:
            print("[Trainer] Running in HIGH-PERFORMANCE PYTORCH mode.")
            print(f"[Trainer] Initializing 1-Bit BitTransformerLM (Vocab: {vocab_size}, Embed: {EMBED_DIM}, Layers: {NUM_LAYERS}, Heads: {NUM_HEADS})...")
            
            model = BitTransformerLM(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                seq_len=SEQ_LEN
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Trainer] Running training on device: {device.type.upper()}")
            model.to(device)
            
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            model.train()
            
            print("[Trainer] Starting Training Loop...")
            step = 0
            total_loss = 0.0
            start_time = time.time()
            
            for epoch in range(1, EPOCHS + 1):
                chunk_gen = self.get_token_chunk_stream()
                batch_gen = self.get_batch_generator(chunk_gen, BATCH_SIZE)
                
                for batch_data in batch_gen:
                    step += 1
                    
                    batch_tensor = torch.tensor(batch_data, dtype=torch.long, device=device)
                    x = batch_tensor[:, :-1]
                    y = batch_tensor[:, 1:]
                    
                    optimizer.zero_grad()
                    logits, loss = model(x, y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if step % 10 == 0 or step == 1:
                        avg_loss = total_loss / (step if step == 1 else 10)
                        tokens_per_sec = (step * BATCH_SIZE * SEQ_LEN) / (time.time() - start_time)
                        print(f"Epoch {epoch} | Step {step:4d} | Loss: {avg_loss:.4f} | Speed: {tokens_per_sec:.1f} tok/sec")
                        total_loss = 0.0
                        
                    if self.limit_steps and step >= self.limit_steps:
                        print(f"[Trainer] Reached step limit: {self.limit_steps}")
                        break
                
                if self.limit_steps and step >= self.limit_steps:
                    break
                    
            print(f"[Trainer] Saving PyTorch checkpoint to {MODEL_CHECKPOINT}")
            torch.save(model.state_dict(), MODEL_CHECKPOINT)
            self.export_to_numpy(model)
            
        else:
            print("\n[Trainer] Running in LOW-POWER PURE-NUMPY mode.")
            print(f"[Trainer] Initializing 1-Bit NumPyBitRNNLM (Vocab: {vocab_size}, Embed/Hidden: {EMBED_DIM})...")
            
            model = NumPyBitRNNLM(vocab_size=vocab_size, embed_dim=EMBED_DIM, seq_len=SEQ_LEN)
            
            print("[Trainer] Starting Training Loop (NumPy backprop with Straight-Through Estimator)...")
            step = 0
            total_loss = 0.0
            start_time = time.time()
            
            for epoch in range(1, EPOCHS + 1):
                chunk_gen = self.get_token_chunk_stream()
                batch_gen = self.get_batch_generator(chunk_gen, BATCH_SIZE)
                
                for batch_data in batch_gen:
                    step += 1
                    
                    # Convert to numpy array
                    batch_np = np.array(batch_data, dtype=np.int32)
                    x = batch_np[:, :-1]
                    y = batch_np[:, 1:]
                    
                    # Run train step (computes forward, BPTT, and Adam updates)
                    loss = model.train_step(x, y, lr=LR)
                    total_loss += loss
                    
                    if step % 10 == 0 or step == 1:
                        avg_loss = total_loss / (step if step == 1 else 10)
                        tokens_per_sec = (step * BATCH_SIZE * SEQ_LEN) / (time.time() - start_time)
                        print(f"Epoch {epoch} | Step {step:4d} | Loss: {avg_loss:.4f} | Speed: {tokens_per_sec:.1f} tok/sec")
                        total_loss = 0.0
                        
                    if self.limit_steps and step >= self.limit_steps:
                        print(f"[Trainer] Reached step limit: {self.limit_steps}")
                        break
                        
                if self.limit_steps and step >= self.limit_steps:
                    break
                    
            print(f"[Trainer] Training completed in {time.time() - start_time:.2f} seconds.")
            
            # Save NumPy weights
            numpy_weight_path = os.path.join(os.path.dirname(MODEL_CHECKPOINT), "numpy_weights.npz")
            np.savez_compressed(
                numpy_weight_path,
                model_type=np.array("rnn"),
                vocab_size=np.array(vocab_size),
                embed_dim=np.array(EMBED_DIM),
                seq_len=np.array(SEQ_LEN),
                E=model.E,
                W_xh=model.W_xh,
                W_hh=model.W_hh,
                W_hy=model.W_hy,
                b_h=model.b_h,
                b_y=model.b_y
            )
            print(f"[Trainer] Pure NumPy 1-bit RNN weights saved successfully to {numpy_weight_path}")
            
    def export_to_numpy(self, model):
        """
        Extracts weights from PyTorch model and saves as compressed NumPy .npz file.
        """
        print("[Trainer] Exporting PyTorch weights to pure NumPy 1-Bit engine...")
        state_dict = model.state_dict()
        
        flat_weights = {
            "model_type": np.array("transformer"),
            "num_heads": np.array(NUM_HEADS),
            "token_embedding": state_dict["token_embedding.weight"].cpu().numpy(),
            "position_embedding": state_dict["position_embedding.weight"].cpu().numpy(),
            "ln_f_w": state_dict["ln_f.weight"].cpu().numpy(),
            "ln_f_b": state_dict["ln_f.bias"].cpu().numpy(),
            "lm_head_w": state_dict["lm_head.weight"].cpu().numpy()
        }
        
        for idx in range(NUM_LAYERS):
            prefix = f"blocks.{idx}."
            flat_weights[f"block_{idx}_ln1_w"] = state_dict[prefix + "ln1.weight"].cpu().numpy()
            flat_weights[f"block_{idx}_ln1_b"] = state_dict[prefix + "ln1.bias"].cpu().numpy()
            flat_weights[f"block_{idx}_ln2_w"] = state_dict[prefix + "ln2.weight"].cpu().numpy()
            flat_weights[f"block_{idx}_ln2_b"] = state_dict[prefix + "ln2.bias"].cpu().numpy()
            
            flat_weights[f"block_{idx}_q_proj_w"] = state_dict[prefix + "attn.q_proj.weight"].cpu().numpy()
            flat_weights[f"block_{idx}_k_proj_w"] = state_dict[prefix + "attn.k_proj.weight"].cpu().numpy()
            flat_weights[f"block_{idx}_v_proj_w"] = state_dict[prefix + "attn.v_proj.weight"].cpu().numpy()
            flat_weights[f"block_{idx}_out_proj_w"] = state_dict[prefix + "attn.out_proj.weight"].cpu().numpy()
            
            flat_weights[f"block_{idx}_gate_proj_w"] = state_dict[prefix + "mlp.gate_proj.weight"].cpu().numpy()
            flat_weights[f"block_{idx}_down_proj_w"] = state_dict[prefix + "mlp.down_proj.weight"].cpu().numpy()
            
        numpy_weight_path = os.path.join(os.path.dirname(MODEL_CHECKPOINT), "numpy_weights.npz")
        np.savez_compressed(numpy_weight_path, **flat_weights)
        print(f"[Trainer] Pure NumPy 1-bit Transformer weights exported successfully to {numpy_weight_path}")
