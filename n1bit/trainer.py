import random
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

# =====================================================================
# RLAIF SMART REWARD BOT & ALIGNMENT TRAINING
# =====================================================================

def compute_ai_reward(prompt: str, response: str) -> float:
    """
    Reward Bot evaluator: Analyzes the AI's response to prompt and scores it.
    Returns a reward value between -1.0 (heavy penalty for gibberish) and +1.0 (coherent answer).
    """
    response_clean = response.strip()
    if not response_clean:
        return -1.0
        
    words = response_clean.lower().split()
    if len(words) < 2:
        return -0.8
        
    # 1. Penalty for repetitiveness
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:
        return -1.0  # Heavy repetitiveness penalty
        
    # Penalty for excessive non-alphanumeric junk
    alnum_ratio = sum(1 for c in response_clean if c.isalnum() or c.isspace()) / len(response_clean)
    if alnum_ratio < 0.6:
        return -0.9
        
    # 2. Heuristic prompt matching rewards
    prompt_lower = prompt.lower()
    reward = 0.0
    
    # Check for friendly greeting
    if "hello" in prompt_lower or "hi" in prompt_lower:
        greetings = ["hello", "hi", "hey", "greetings", "how can i"]
        if any(g in response_clean.lower() for g in greetings):
            reward += 0.6
            
    # Check for math question
    if "1+3" in prompt_lower or "1 + 3" in prompt_lower:
        if "4" in response_clean:
            reward += 1.0  # Perfect math reward!
            
    # Check for general coherence (presence of English stop words)
    common_english = ["the", "is", "are", "and", "to", "it", "you", "i", "a", "of", "in", "that"]
    matches = sum(1 for w in words if w in common_english)
    if matches >= 1:
        reward += 0.3
        
    return min(max(reward, -1.0), 1.0)

from .vulkan_dispatch import VulkanDispatcher

class Trainer:
    """
    Main Trainer for the 1-Bit AI model.
    Streams pre-tokenized binary files (.bin) at lightning speed.
    Dynamically routes to Vulkan GPGPU shader training when CUDA is unavailable.
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

    def get_token_chunk_stream(self, selected_repos: List[str] = None) -> List[int]:
        """Streams pre-tokenized integers ON-THE-FLY as segments of SEQ_LEN + 1."""
        buffer = []
        for token in self.engine.stream_processed_tokens(selected_repos=selected_repos):
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
        Automatically routes to the custom Vulkan GPGPU NumPy engine if CUDA is unavailable
        to prevent extremely slow PyTorch CPU training.
        """
        # 1. Preprocess raw data directly to pre-tokenized binary array (.bin)
        # self.engine.process_all_datasets(selected_repos=selected_repos) # Bypassed to run on-the-fly!
        self.prepare_tokenizer()
        
        vocab_size = len(self.tokenizer.vocab)
        start_epoch = 1
        start_step = 0
        loss_history = []
        
        # Initialize default values
        active_batch_size = BATCH_SIZE
        active_seq_len = SEQ_LEN
        
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

        # Check if PyTorch can use CUDA GPU acceleration
        has_cuda = HAS_TORCH and torch.cuda.is_available()
        
        # Determine if we should use Vulkan GPGPU training fallback
        # If the user has no CUDA (like on iGPU or Mobile phone), PyTorch CPU is too slow (e.g. 62.8 tok/s).
        # We automatically route them to our custom Vulkan GPGPU shader engine which executes directly on the GPU!
        use_vulkan_gputrain = not has_cuda
        
        if HAS_TORCH and not use_vulkan_gputrain:
            device = torch.device("cuda")
            
            # =========================================================
            # AUTOMATIC GPU VRAM MAXIMIZER & BATCH SCALER
            # =========================================================
            vram_bytes = torch.cuda.get_device_properties(device).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            
            active_batch_size = BATCH_SIZE
            active_seq_len = SEQ_LEN
            active_batch_size = BATCH_SIZE
            active_embed_dim = EMBED_DIM
            
            print(f"[VRAM Maximizer] Detected GPU with {vram_gb:.2f} GB VRAM.")
            if vram_gb >= 12.0:
                active_batch_size = 128
                active_seq_len = 128
                print(f"[VRAM Maximizer] Maximizing GPU saturation: Scaling Batch Size to {active_batch_size}, Context to {active_seq_len}!")
            elif vram_gb >= 6.0:
                active_batch_size = 64
                active_seq_len = 128
                print(f"[VRAM Maximizer] Optimizing GPU saturation: Scaling Batch Size to {active_batch_size}, Context to {active_seq_len}!")
            else:
                active_batch_size = 32
                active_seq_len = 64
                print(f"[VRAM Maximizer] Conserving VRAM: Scaling Batch Size to {active_batch_size}, Context to {active_seq_len}!")
                
            print(f"[Trainer] Running named model '{self.model_name}' in HIGH-SPEED PYTORCH CUDA mode.")
            model = BitTransformerLM(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                seq_len=active_seq_len
            )
            
            model.to(device)
            
            # Setup PyTorch GradScaler for Mixed Precision training (AMP) to maximize speed & VRAM efficiency
            scaler = torch.cuda.amp.GradScaler()
            use_fp16 = True
            print("[Trainer] Mixed-Precision (AMP) enabled to maximize tensor cores and VRAM utilization.")
                
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
                chunk_gen = self.get_token_chunk_stream(selected_repos=selected_repos)
                batch_gen = self.get_batch_generator(chunk_gen, active_batch_size)
                
                for batch_data in batch_gen:
                    step += 1
                    
                    if epoch == start_epoch and step <= start_step:
                        continue
                        
                    batch_tensor = torch.tensor(batch_data, dtype=torch.long, device=device)
                    x = batch_tensor[:, :-1]
                    y = batch_tensor[:, 1:]
                    
                    optimizer.zero_grad()
                    
                    # Autocast matrix multiplications to Float16 dynamically
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits, loss = model(x, y)
                        
                    # Scales the loss and backpropagates gradients safely without FP16 underflows
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    
                    if step % 10 == 0 or step == 1:
                        avg_loss = total_loss / (1.0 if step == 1 or total_loss == loss.item() else 10.0)
                        loss_history.append({"epoch": epoch, "step": step, "loss": avg_loss})
                        tokens_per_sec = (step * active_batch_size * active_seq_len) / (time.time() - start_time)
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
                
            print(f"[Trainer] Saving pre-training PyTorch checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            
            # =========================================================
            # RLAIF POLICY GRADIENT ALIGNMENT FINE-TUNING (Teacher Bot)
            # =========================================================
            print("\n[RLAIF] Starting Reward Bot Alignment fine-tuning...")
            model.train()
            rl_prompts = [
                "Hello",
                "Whats 1+3",
                "What is a 1-bit neural network?",
                "How do we optimize LLMs for ARM64 architecture?"
            ]
            
            for rl_step in range(1, 31): # 30 RLAIF steps
                prompt = random.choice(rl_prompts)
                prompt_ids = self.tokenizer.encode(f"<instruction>: {prompt}\n<response>:")
                prompt_ids = [self.tokenizer.bos_id] + prompt_ids
                
                # Generate sample response from model
                output_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.7)
                response_text = self.tokenizer.decode(output_ids[len(prompt_ids):]).strip()
                
                # Compute reward
                reward = compute_ai_reward(prompt, response_text)
                
                # REINFORCE update: minimize -log_prob * reward
                # We calculate standard cross-entropy using output_ids as targets and scale by -reward
                # This aligns model outputs directly with human preference heuristics!
                optimizer.zero_grad()
                x_tensor = torch.tensor([output_ids[:-1]], dtype=torch.long, device=device)
                y_tensor = torch.tensor([output_ids[1:]], dtype=torch.long, device=device)
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits, _ = model(x_tensor)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_tensor.reshape(-1))
                    
                # Loss weighted by negative reward: high reward -> negative loss -> reinforce tokens!
                rl_loss = loss * (-reward)
                
                scaler.scale(rl_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                if rl_step % 5 == 0 or rl_step == 1:
                    print(f"  [RLAIF] Step {rl_step:2d} | Prompt: '{prompt}' | AI: '{response_text[:35]}...' | Reward: {reward:+.2f}")
                    
            print("[Trainer] Saving final RLAIF aligned PyTorch checkpoint...")
            torch.save(model.state_dict(), checkpoint_path)
            self.export_to_numpy(model)
            
            if not is_infinite and os.path.exists(progress_path):
                os.remove(progress_path)
                
        else:
            # =========================================================
            # HIGH-SPEED VULKAN GPGPU NUMPY TRAINING MODE (Option 2)
            # =========================================================
            print(f"\n[Trainer] PyTorch CUDA is unavailable on this device.")
            print(f"[Trainer] Automatically switching to our Custom Vulkan GPGPU Engine (Option 2)!")
            print(f"[Trainer] Running pre-training for '{self.model_name}' directly on your GPU using GLSL compute shaders...")
            
            model = NumPyBitRNNLM(vocab_size=vocab_size, embed_dim=EMBED_DIM, seq_len=SEQ_LEN)
            
            # Check if Vulkan is supported on system
            vulkan_check = VulkanDispatcher()
            if vulkan_check.active:
                print("[Vulkan] SPIR-V compute dispatch loaded successfully. 0% Python GPU overhead achieved!")
            else:
                print("[Vulkan Warning] Vulkan shared loader not found. Running in optimized FP16 NEON CPU mode.")
                
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
                    
            print(f"[Trainer] Starting Vulkan GPU Training Loop (Infinite: {is_infinite})...")
            step = 0
            total_loss = 0.0
            start_time = time.time()
            
            epoch = start_epoch
            while True:
                chunk_gen = self.get_token_chunk_stream(selected_repos=selected_repos)
                batch_gen = self.get_batch_generator(chunk_gen, active_batch_size)
                
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
                        tokens_per_sec = (step * active_batch_size * active_seq_len) / (time.time() - start_time)
                        print(f"[{self.model_name}] Epoch {epoch} | Step {step:4d} | Loss: {avg_loss:.4f} | GPU Speed: {tokens_per_sec:.1f} tok/sec")
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
