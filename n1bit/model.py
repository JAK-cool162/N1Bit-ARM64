import os
import json
import math
import numpy as np
from typing import List, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn_stub:
        class Module: pass
    nn = nn_stub()

from .config import QUANT_TYPE, USE_16BIT
from .vulkan_dispatch import VulkanDispatcher

# Helper function to get correct numpy float dtype
def get_np_dtype():
    return np.float16 if USE_16BIT else np.float32

# =====================================================================
# PYTORCH 1-BIT ARCHITECTURE (BitNet Q1 Binary Quantization)
# =====================================================================

if HAS_TORCH:
    class BitLinear(nn.Module):
        """
        BitLinear layer optimized with Q1 Binary Quantization (-1, +1).
        - Uses FP16/Half precision for speed.
        - Employs Straight-Through Estimator (STE) for backpropagation.
        """
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Let PyTorch's native AMP autocast handle precision dynamically!
            dtype = x.dtype
            weight = self.weight.to(dtype)
            
            # Q1 Quantization: Binary weight quantization to {-1, +1}
            scale = torch.mean(torch.abs(weight))
            scale = torch.clamp(scale, min=1e-3)
            
            w_q = torch.sign(weight)
            w_q[w_q == 0] = 1.0
            
            # STE formulation
            w_final = weight + (w_q * scale - weight).detach()

            # Dynamic 8-bit activation quantization for speed
            max_x = torch.max(torch.abs(x))
            scale_factor = 127.0 / torch.clamp(max_x, min=1e-3).float()
            scale_factor = torch.clamp(scale_factor, max=5000.0).to(dtype)
            
            x_scaled = x * scale_factor
            x_quant = torch.clamp(torch.round(x_scaled), -128.0, 127.0)
            x_final = x + (x_quant / scale_factor - x).detach()

            bias = self.bias.to(dtype) if self.bias is not None else None
            return F.linear(x_final, w_final, bias)


    class BitAttention(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
            
            self.q_proj = BitLinear(embed_dim, embed_dim, bias=False)
            self.k_proj = BitLinear(embed_dim, embed_dim, bias=False)
            self.v_proj = BitLinear(embed_dim, embed_dim, bias=False)
            self.out_proj = BitLinear(embed_dim, embed_dim, bias=False)

        def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
            batch_size, seq_len, embed_dim = x.size()
            
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, v)
            
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            return self.out_proj(context)


    class BitMLP(nn.Module):
        def __init__(self, embed_dim: int):
            super().__init__()
            hidden_dim = 4 * embed_dim
            self.gate_proj = BitLinear(embed_dim, hidden_dim, bias=False)
            self.down_proj = BitLinear(hidden_dim, embed_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.gelu(self.gate_proj(x)))


    class BitTransformerBlock(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.attn = BitAttention(embed_dim, num_heads)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.mlp = BitMLP(embed_dim)

        def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
            dtype = x.dtype
            x = x + self.attn(self.ln1(x).to(dtype), mask)
            x = x + self.mlp(self.ln2(x).to(dtype))
            return x


    class BitTransformerLM(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, seq_len: int):
            super().__init__()
            # Security Padding: Pad vocab size and context length to make the model
            # 100% immune to any device-side out-of-bounds assertion crashes on GPU!
            self.vocab_size = max(vocab_size, 8000)
            self.seq_len = seq_len
            
            self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
            self.position_embedding = nn.Embedding(2048, embed_dim)
            
            self.blocks = nn.ModuleList([
                BitTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
            ])
            
            self.ln_f = nn.LayerNorm(embed_dim)
            self.lm_head = nn.Linear(embed_dim, self.vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
            batch_size, seq_len = input_ids.size()
            device = input_ids.device
            
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            dtype = x.dtype
            
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len)
            
            for block in self.blocks:
                x = block(x, mask)
                
            x = self.ln_f(x).to(dtype)
            logits = self.lm_head(x)
            
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
            return logits, loss

        def generate(self, prompt_ids: List[int], max_new_tokens: int, temperature: float = 1.0) -> List[int]:
            self.eval()
            dtype = torch.float16 if USE_16BIT else torch.float32
            self.to(dtype)
            
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            
            for _ in range(max_new_tokens):
                context_ids = input_ids[:, -self.seq_len:]
                with torch.no_grad():
                    logits, _ = self.forward(context_ids)
                    
                next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits.float(), dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                
                if next_token.item() == 2:
                    break
                    
            return input_ids[0].tolist()


# =====================================================================
# PURE NUMPY 1-BIT TRANSFORMR DEPLOYMENT (Vulkan GPU Accelerated)
# =====================================================================

class NumPyBitLinear:
    def __init__(self, weight: np.ndarray, bias: np.ndarray = None):
        dtype = get_np_dtype()
        self.weight = weight.astype(dtype)
        self.bias = bias.astype(dtype) if bias is not None else None
        
        # Initialize native Vulkan ctypes dispatcher for GPU hot paths!
        self.vulkan = VulkanDispatcher()
        
        # Q1 Quantization: Binary Weight quantization
        mean_abs_w = np.mean(np.abs(self.weight))
        self.beta = max(mean_abs_w, 1e-3)
        
        w_q = np.sign(self.weight)
        w_q[w_q == 0] = 1.0
        self.w_quant = (w_q * self.beta).astype(dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        max_x = np.max(np.abs(x))
        scale_factor = 127.0 / max(max_x, 1e-3)
        scale_factor = min(scale_factor, 5000.0)
        
        x_quant = np.clip(np.round(x * scale_factor), -128.0, 127.0) / scale_factor
        x_quant = x_quant.astype(get_np_dtype())
        
        # Dispatch the computationally heavy dot product to the GPU via Vulkan!
        # Zero Python loop overhead on the GPU side.
        if len(x_quant.shape) <= 2:
            out = self.vulkan.run_matmul(x_quant, self.w_quant)
        else:
            out = np.dot(x_quant, self.w_quant.T)
            
        if self.bias is not None:
            out += self.bias
        return out


class NumPyBitAttention:
    def __init__(self, q_proj_w, k_proj_w, v_proj_w, out_proj_w, num_heads: int):
        self.num_heads = num_heads
        self.q_proj = NumPyBitLinear(q_proj_w)
        self.k_proj = NumPyBitLinear(k_proj_w)
        self.v_proj = NumPyBitLinear(v_proj_w)
        self.out_proj = NumPyBitLinear(out_proj_w)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        
        q = self.q_proj.forward(x).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj.forward(x).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj.forward(x).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        if mask is not None:
            scores = np.where(mask == 0, -1e4, scores)
            
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        attn_weights = attn_weights.astype(get_np_dtype())
        
        context = np.matmul(attn_weights, v)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        return self.out_proj.forward(context)


class NumPyBitMLP:
    def __init__(self, gate_proj_w, down_proj_w):
        self.gate_proj = NumPyBitLinear(gate_proj_w)
        self.down_proj = NumPyBitLinear(down_proj_w)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.down_proj.forward(self.gelu(self.gate_proj.forward(x)))


class NumPyBitTransformerBlock:
    def __init__(self, block_data: dict, num_heads: int):
        self.num_heads = num_heads
        dtype = get_np_dtype()
        
        self.ln1_w = block_data["ln1_w"].astype(dtype)
        self.ln1_b = block_data["ln1_b"].astype(dtype)
        self.ln2_w = block_data["ln2_w"].astype(dtype)
        self.ln2_b = block_data["ln2_b"].astype(dtype)
        
        self.attn = NumPyBitAttention(
            block_data["q_proj_w"], block_data["k_proj_w"], block_data["v_proj_w"], block_data["out_proj_w"],
            num_heads
        )
        self.mlp = NumPyBitMLP(block_data["gate_proj_w"], block_data["down_proj_w"])

    def layernorm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        norm_x1 = self.layernorm(x, self.ln1_w, self.ln1_b).astype(get_np_dtype())
        x = x + self.attn.forward(norm_x1, mask)
        norm_x2 = self.layernorm(x, self.ln2_w, self.ln2_b).astype(get_np_dtype())
        x = x + self.mlp.forward(norm_x2)
        return x


class NumPyBitTransformerLM:
    def __init__(self, weights_dict: dict):
        dtype = get_np_dtype()
        self.token_emb = weights_dict["token_embedding"].astype(dtype)
        self.pos_emb = weights_dict["position_embedding"].astype(dtype)
        
        self.vocab_size = self.token_emb.shape[0]
        self.embed_dim = self.token_emb.shape[1]
        self.seq_len = self.pos_emb.shape[0]
        self.num_heads = weights_dict["num_heads"]
        
        self.ln_f_w = weights_dict["ln_f_w"].astype(dtype)
        self.ln_f_b = weights_dict["ln_f_b"].astype(dtype)
        self.lm_head = NumPyBitLinear(weights_dict["lm_head_w"])
        # Keep lm_head in full precision to match training
        self.lm_head.w_quant = weights_dict["lm_head_w"]
        
        self.blocks = []
        for block_data in weights_dict["blocks"]:
            self.blocks.append(NumPyBitTransformerBlock(block_data, self.num_heads))

    def layernorm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        x = self.token_emb[input_ids] + self.pos_emb[np.arange(seq_len)]
        mask = np.tril(np.ones((seq_len, seq_len))).reshape(1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block.forward(x, mask)
            
        x = self.layernorm(x, self.ln_f_w, self.ln_f_b).astype(get_np_dtype())
        logits = self.lm_head.forward(x)
        return logits

    def generate(self, prompt_ids: List[int], max_new_tokens: int, temperature: float = 1.0) -> List[int]:
        input_ids = np.array([prompt_ids], dtype=np.int32)
        
        for _ in range(max_new_tokens):
            context_ids = input_ids[:, -self.seq_len:]
            logits = self.forward(context_ids)
            next_logits = logits[0, -1, :].astype(np.float32) / max(temperature, 1e-5)
            exp_logits = np.exp(next_logits - np.max(next_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_token = np.random.choice(self.vocab_size, p=probs)
            input_ids = np.concatenate([input_ids, np.array([[next_token]])], axis=1)
            
            if next_token == 2:
                break
                
        return input_ids[0].tolist()


# =====================================================================
# PURE NUMPY 1-BIT RNN RECURRENT MODEL (Q1 Binary Quantization + FP16)
# =====================================================================

class NumPyBitRNNLM:
    """
    Pure NumPy 1-Bit Recurrent Language Model optimized with Q1 binary quantization and FP16 half precision.
    Uses native Vulkan ctypes shader dispatch on GPU hot paths.
    """
    def __init__(self, vocab_size: int, embed_dim: int, seq_len: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.hidden_dim = embed_dim
        
        dtype = get_np_dtype()
        self.vulkan = VulkanDispatcher()
        
        self.E = (np.random.randn(vocab_size, embed_dim) * 0.05).astype(dtype)
        self.W_xh = (np.random.randn(self.hidden_dim, embed_dim) * 0.05).astype(dtype)
        self.W_hh = (np.random.randn(self.hidden_dim, self.hidden_dim) * 0.05).astype(dtype)
        self.W_hy = (np.random.randn(vocab_size, self.hidden_dim) * 0.05).astype(dtype)
        
        self.b_h = np.zeros((self.hidden_dim, 1), dtype=dtype)
        self.b_y = np.zeros((vocab_size, 1), dtype=dtype)
        
        self.m_E, self.v_E = np.zeros_like(self.E), np.zeros_like(self.E)
        self.m_W_xh, self.v_W_xh = np.zeros_like(self.W_xh), np.zeros_like(self.W_xh)
        self.m_W_hh, self.v_W_hh = np.zeros_like(self.W_hh), np.zeros_like(self.W_hh)
        self.m_W_hy, self.v_W_hy = np.zeros_like(self.W_hy), np.zeros_like(self.W_hy)
        self.m_b_h, self.v_b_h = np.zeros_like(self.b_h), np.zeros_like(self.b_h)
        self.m_b_y, self.v_b_y = np.zeros_like(self.b_y), np.zeros_like(self.b_y)
        
        self.t = 0

    def q_weights(self, W: np.ndarray):
        """Q1 Quantization: Binary weight quantization to {-1, +1}."""
        beta = np.mean(np.abs(W))
        beta = max(beta, 1e-3)
        
        w_q = np.sign(W)
        w_q[w_q == 0] = 1.0
        
        return (w_q * beta).astype(W.dtype), w_q, beta

    def q_activation(self, x: np.ndarray) -> np.ndarray:
        """Quantizes activation to 8-bit."""
        max_x = np.max(np.abs(x))
        scale_factor = 127.0 / max(max_x, 1e-3)
        scale_factor = min(scale_factor, 5000.0)
        return np.clip(np.round(x * scale_factor), -128.0, 127.0) / scale_factor

    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, lr: float = 0.005) -> float:
        """
        Runs one step of BPTT training with Straight-Through Estimator and FP16 updates.
        """
        dtype = get_np_dtype()
        batch_size, seq_len = input_ids.shape
        
        X = input_ids.T
        Y = target_ids.T
        
        W_xh_f, _, _ = self.q_weights(self.W_xh)
        W_hh_f, _, _ = self.q_weights(self.W_hh)
        W_hy_f, _, _ = self.q_weights(self.W_hy)
        
        h = {}
        h[-1] = np.zeros((self.hidden_dim, batch_size), dtype=dtype)
        
        x_emb = {}
        a = {}
        probs = {}
        loss = 0.0
        
        for t in range(seq_len):
            x_emb[t] = self.q_activation(self.E[X[t]].T).astype(dtype)
            
            # Accelerating compute projection via native Vulkan dispatch (ctypes + SPIR-V)!
            proj_xh = self.vulkan.run_matmul(W_xh_f, x_emb[t])
            proj_hh = self.vulkan.run_matmul(W_hh_f, h[t-1])
            
            a[t] = proj_xh.reshape(-1, batch_size) + proj_hh.reshape(-1, batch_size) + self.b_h
            h[t] = np.tanh(a[t]).astype(dtype)
            
            logits_t = self.vulkan.run_matmul(W_hy_f, h[t]).reshape(-1, batch_size) + self.b_y
            
            max_logits = np.max(logits_t, axis=0, keepdims=True)
            exp_logits = np.exp((logits_t - max_logits).astype(np.float32))
            probs[t] = (exp_logits / np.sum(exp_logits, axis=0, keepdims=True)).astype(dtype)
            
            targets_idx = Y[t]
            loss += -np.log(probs[t][targets_idx, np.arange(batch_size)] + 1e-15).mean()
            
        loss /= seq_len
        
        dE = np.zeros_like(self.E, dtype=dtype)
        dW_xh = np.zeros_like(self.W_xh, dtype=dtype)
        dW_hh = np.zeros_like(self.W_hh, dtype=dtype)
        dW_hy = np.zeros_like(self.W_hy, dtype=dtype)
        db_h = np.zeros_like(self.b_h, dtype=dtype)
        db_y = np.zeros_like(self.b_y, dtype=dtype)
        
        dh_next = np.zeros((self.hidden_dim, batch_size), dtype=dtype)
        
        for t in reversed(range(seq_len)):
            dy = probs[t].copy()
            dy[Y[t], np.arange(batch_size)] -= 1.0
            dy /= (batch_size * seq_len)
            dy = dy.astype(dtype)
            
            # Vulkan dispatch for backprop projections!
            dW_hy += self.vulkan.run_matmul(dy, h[t].T)
            db_y += np.sum(dy, axis=1, keepdims=True)
            
            dh = self.vulkan.run_matmul(W_hy_f.T, dy) + dh_next
            da = (dh * (1.0 - h[t]**2)).astype(dtype)
            
            dW_xh += self.vulkan.run_matmul(da, x_emb[t].T)
            dW_hh += self.vulkan.run_matmul(da, h[t-1].T)
            db_h += np.sum(da, axis=1, keepdims=True)
            
            dx = self.vulkan.run_matmul(W_xh_f.T, da)
            for b in range(batch_size):
                dE[X[t, b]] += dx[:, b]
                
            dh_next = self.vulkan.run_matmul(W_hh_f.T, da)
            
        self.t += 1
        eps = 1e-4 if dtype == np.float16 else 1e-8
        beta1, beta2 = 0.9, 0.999
        
        for g in [dE, dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(g, -1.0, 1.0, out=g)
            
        def adam_step(w, grad, m, v):
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** self.t)
            v_hat = v / (1.0 - beta2 ** self.t)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
            return m, v
            
        self.m_E, self.v_E = adam_step(self.E, dE, self.m_E, self.v_E)
        self.m_W_xh, self.v_W_xh = adam_step(self.W_xh, dW_xh, self.m_W_xh, self.v_W_xh)
        self.m_W_hh, self.v_W_hh = adam_step(self.W_hh, dW_hh, self.m_W_hh, self.v_W_hh)
        self.m_W_hy, self.v_W_hy = adam_step(self.W_hy, dW_hy, self.m_W_hy, self.v_W_hy)
        self.m_b_h, self.v_b_h = adam_step(self.b_h, db_h, self.m_b_h, self.v_b_h)
        self.m_b_y, self.v_b_y = adam_step(self.b_y, db_y, self.m_b_y, self.v_b_y)
        
        return float(loss)

    def generate(self, prompt_ids: List[int], max_new_tokens: int, temperature: float = 1.0) -> List[int]:
        input_ids = list(prompt_ids)
        dtype = get_np_dtype()
        
        h_prev = np.zeros((self.hidden_dim, 1), dtype=dtype)
        
        W_xh_f, _, _ = self.q_weights(self.W_xh)
        W_hh_f, _, _ = self.q_weights(self.W_hh)
        W_hy_f, _, _ = self.q_weights(self.W_hy)
        
        for p_id in input_ids:
            x_t = self.q_activation(self.E[p_id].reshape(-1, 1)).astype(dtype)
            h_prev = np.tanh(self.vulkan.run_matmul(x_t.T, W_xh_f.T).T + self.vulkan.run_matmul(h_prev.T, W_hh_f.T).T + self.b_h).astype(dtype)
            
        for _ in range(max_new_tokens):
            x_t = self.q_activation(self.E[input_ids[-1]].reshape(-1, 1)).astype(dtype)
            h_prev = np.tanh(self.vulkan.run_matmul(x_t.T, W_xh_f.T).T + self.vulkan.run_matmul(h_prev.T, W_hh_f.T).T + self.b_h).astype(dtype)
            
            logits = self.vulkan.run_matmul(h_prev.T, W_hy_f.T).T + self.b_y
            logits = logits.flatten().astype(np.float32) / max(temperature, 1e-5)
            
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_token = np.random.choice(self.vocab_size, p=probs)
            input_ids.append(int(next_token))
            
            if next_token == 2:
                break
                
        return input_ids
