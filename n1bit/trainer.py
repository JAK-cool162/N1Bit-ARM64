import os
import json
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
    BATCH_SIZE, SEQ_LEN, EMBED_DIM, NUM_LAYERS, NUM_HEADS, LR, EPOCHS, USE_16BIT,
    get_model_paths
)
from .utils import optimize_environment
from .tokenizer import SimpleBPETokenizer
from .dataset import DatasetEngine
from .model import NumPyBitRNNLM

class Trainer:
    """
    Main Trainer for the 1-Bit AI model.
    Streams pre-tokenized binary files (.bin) at lightning speed.
    """
    def __init__(self, model_name: str = "default", limit_steps=None):
        self.model_name = model_name
        
        if limit_steps == "inf" or limit_steps == float('inf'):
            self.limit_steps = float('inf')
        elif limit_steps is not None:
            self.limit_steps = int(limit_steps)
        else:
            self.limit_steps = None
            
        self.paths = get_model_paths(self.model_name)
        
        # Initialize DatasetEngine with model-isolated paths and tokenizer file
        self.engine = DatasetEngine(self.paths["processed_data"], self.paths["stats"], self.paths["tokenizer"])
        self.tokenizer = SimpleBPETokenizer(vocab_size=4000)
        
        optimize_environment()
        
    def prepare_tokenizer(self):
        """Loads the tokenizer inside the model's directory."""
        tokenizer_path = self.paths["tokenizer"]
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)

    def get_token_chunk_stream(self) -> List[int]:
        """Streams pre-tokenized integers directly from the binary file as segments of SEQ_LEN + 1."""
        buffer = []
        for token in self.engine.stream_processed_tokens():
            buffer.append(token)
            if len(buffer) >= SEQ_LEN + 1:
                yield buffer[:SEQ_LEN + 1]
                buffer = buffer[SEQ_LEN + 1:]

    def get_batch_generator(self, chunk_generator, batch_size: int):
        """Groups sequence chunks into batches."""
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

    def train(self, selected_repos: List[str] = None):
        """
        Trains the named 1-bit model.
        """
        # 1. Preprocess raw data directly to pre-tokenized binary array (.bin)
        self.engine.process_all_datasets(selected_repos=selected_repos)
        self.prepare_tokenizer()
        
        vocab_size = len(self.tokenizer.vocab)
        start_epoch = 1
        start_step = 0
        loss_history = []
        
        # Load progress if it exists
        resumed = False
        progress_path = self.paths["progress"]
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)
                start_epoch = progress_data.get("epoch", 1)
                start_step = progress_data.get("step", 0)
                loss_history = progress_data.get("loss_history", [])
                print(f"\n[Trainer] Found existing progress for model '{self.model_name}'! Resuming from Epoch {start_epoch}, Step {start_step}...")
                resumed = True
            except Exception as e:
                print(f"[Trainer] Failed to load progress JSON: {e}. Starting fresh.")

        is_infinite = (self.limit_steps == float('inf'))
        target_epochs = 100000 if is_infinite else EPOCHS
        
        checkpoint_path = self.paths["checkpoint"]
        numpy_weight_path = self.paths["numpy_weights"]

        if HAS_TORCH:
            print(f"[Trainer] Running named model '{self.model_name}' in PYTORCH mode.")
            model = BitTransformerLM(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                seq_len=SEQ_LEN
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            use_fp16 = USE_16BIT and device.type == "cuda"
            if use_fp16:
                model = model.half()
                print("[Trainer] CUDA detected. Enabling Float16 precision for training acceleration.")
            else:
                print("[Trainer] Running in standard float32 on CPU for maximum numeric stability.")
                
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            model.train()
            
            if resumed and os.path.exists(checkpoint_path):
                try:
                    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                    print(f"[Trainer] Loaded model '{self.model_name}' weights from checkpoint.")
                except Exception as e:
                    print(f"[Trainer] Could not load weights: {e}. Starting fresh.")
                    
            print(f"[Trainer] Starting PyTorch Training Loop (Infinite: {is_infinite})...")
            step = 0
            total_loss = 0.0
            start_time = time.time()
            
            epoch = start_epoch
            while True:
                chunk_gen = self.get_token_chunk_stream()
                batch_gen = self.get_batch_generator(chunk_gen, BATCH_SIZE)
                
                for batch_data in batch_gen:
                    step += 1
                    
                    if epoch == start_epoch and step <= start_step:
                        continue
                        
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
                        avg_loss = total_loss / (1.0 if step == 1 or total_loss == loss.item() else 10.0)
                        loss_history.append({"epoch": epoch, "step": step, "loss": avg_loss})
                        tokens_per_sec = (step * BATCH_SIZE * SEQ_LEN) / (time.time() - start_time)
                        print(f"[{self.model_name}] Epoch {epoch} | Step {step:4d} | Loss: {avg_loss:.4f} | Speed: {tokens_per_sec:.1f} tok/sec")
                        total_loss = 0.0
                        
                        self.save_progress(epoch, step, loss_history)
                        torch.save(model.state_dict(), checkpoint_path)
                        self.export_to_numpy(model)
                        
                    if not is_infinite and self.limit_steps and step >= self.limit_steps:
                        print(f"[Trainer] Reached step limit: {self.limit_steps}")
                        break
                        
                if not is_infinite and self.limit_steps and step >= self.limit_steps:
                    break
                    
                epoch += 1
                if not is_infinite and epoch > target_epochs:
                    break
                start_step = 0
                
            print(f"[Trainer] Saving final PyTorch checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            self.export_to_numpy(model)
            
            if not is_infinite and os.path.exists(progress_path):
                os.remove(progress_path)
                
        else:
            print(f"\n[Trainer] Running named model '{self.model_name}' in PURE-NUMPY RNN mode.")
            model = NumPyBitRNNLM(vocab_size=vocab_size, embed_dim=EMBED_DIM, seq_len=SEQ_LEN)
            
            if resumed and os.path.exists(numpy_weight_path):
                try:
                    data = np.load(numpy_weight_path, allow_pickle=True)
                    model.E = data["E"]
                    model.W_xh = data["W_xh"]
                    model.W_hh = data["W_hh"]
                    model.W_hy = data["W_hy"]
                    model.b_h = data["b_h"]
                    model.b_y = data["b_y"]
                    
                    model.m_E, model.v_E = data["m_E"], data["v_E"]
                    model.m_W_xh, model.v_W_xh = data["m_W_xh"], data["v_W_xh"]
                    model.m_W_hh, model.v_W_hh = data["m_W_hh"], data["v_W_hh"]
                    model.m_W_hy, model.v_W_hy = data["m_W_hy"], data["v_W_hy"]
                    model.m_b_h, model.v_b_h = data["m_b_h"], data["v_b_h"]
                    model.m_b_y, model.v_b_y = data["m_b_y"], data["v_b_y"]
                    model.t = int(data["optimizer_t"])
                    print(f"[Trainer] Loaded model '{self.model_name}' weights and Adam states.")
                except Exception as e:
                    print(f"[Trainer] Could not load NumPy checkpoints: {e}. Starting fresh.")
                    
            print(f"[Trainer] Starting NumPy Training Loop (Infinite: {is_infinite})...")
            step = 0
            total_loss = 0.0
            start_time = time.time()
            
            epoch = start_epoch
            while True:
                chunk_gen = self.get_token_chunk_stream()
                batch_gen = self.get_batch_generator(chunk_gen, BATCH_SIZE)
                
                for batch_data in batch_gen:
                    step += 1
                    
                    if epoch == start_epoch and step <= start_step:
                        continue
                        
                    batch_np = np.array(batch_data, dtype=np.int32)
                    x = batch_np[:, :-1]
                    y = batch_np[:, 1:]
                    
                    loss = model.train_step(x, y, lr=LR)
                    total_loss += loss
                    
                    if step % 10 == 0 or step == 1:
                        avg_loss = total_loss / (1.0 if step == 1 or total_loss == loss else 10.0)
                        loss_history.append({"epoch": epoch, "step": step, "loss": avg_loss})
                        tokens_per_sec = (step * BATCH_SIZE * SEQ_LEN) / (time.time() - start_time)
                        print(f"[{self.model_name}] Epoch {epoch} | Step {step:4d} | Loss: {avg_loss:.4f} | Speed: {tokens_per_sec:.1f} tok/sec")
                        total_loss = 0.0
                        
                        self.save_progress(epoch, step, loss_history)
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
                            b_y=model.b_y,
                            m_E=model.m_E, v_E=model.v_E,
                            m_W_xh=model.m_W_xh, v_W_xh=model.v_W_xh,
                            m_W_hh=model.m_W_hh, v_W_hh=model.v_W_hh,
                            m_W_hy=model.m_W_hy, v_W_hy=model.v_W_hy,
                            m_b_h=model.m_b_h, v_b_h=model.v_b_h,
                            m_b_y=model.m_b_y, v_b_y=model.v_b_y,
                            optimizer_t=np.array(model.t)
                        )
                        
                    if not is_infinite and self.limit_steps and step >= self.limit_steps:
                        print(f"[Trainer] Reached step limit: {self.limit_steps}")
                        break
                        
                if not is_infinite and self.limit_steps and step >= self.limit_steps:
                    break
                
                epoch += 1
                if not is_infinite and epoch > target_epochs:
                    break
                start_step = 0
                
            print(f"[Trainer] Training completed in {time.time() - start_time:.2f} seconds.")
            
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
                b_y=model.b_y,
                m_E=model.m_E, v_E=model.v_E,
                m_W_xh=model.m_W_xh, v_W_xh=model.v_W_xh,
                m_W_hh=model.m_W_hh, v_W_hh=model.v_W_hh,
                m_W_hy=model.m_W_hy, v_W_hy=model.v_W_hy,
                m_b_h=model.m_b_h, v_b_h=model.v_b_h,
                m_b_y=model.m_b_y, v_b_y=model.v_b_y,
                optimizer_t=np.array(model.t)
            )
            print(f"[Trainer] Pure NumPy 1-bit RNN weights saved successfully to {numpy_weight_path}")
            
            if not is_infinite and os.path.exists(progress_path):
                os.remove(progress_path)

    def save_progress(self, epoch: int, step: int, loss_history: List[dict]):
        """Saves current training progression state to progress JSON file."""
        data = {
            "epoch": epoch,
            "step": step,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "loss_history": loss_history
        }
        with open(self.paths["progress"], 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def export_to_numpy(self, model):
        """Extracts weights from PyTorch model and saves as compressed NumPy .npz file."""
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
            
        numpy_weight_path = self.paths["numpy_weights"]
        np.savez_compressed(numpy_weight_path, **flat_weights)
        print(f"[Trainer] Pure NumPy 1-bit Transformer weights exported successfully to {numpy_weight_path}")
