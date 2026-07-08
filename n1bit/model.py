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
    # Define a stub nn.Module so imports don't fail
    class nn_stub:
        class Module: pass
    nn = nn_stub()

# =====================================================================
# PYTORCH 1-BIT ARCHITECTURE (BitNet b1.58)
# =====================================================================

if HAS_TORCH:
    class BitLinear(nn.Module):
        """
        BitLinear layer as described in BitNet b1.58.
        - Quantizes weights to ternary values (-1, 0, +1) with scaling factor beta.
        - Quantizes activations to 8-bit with scaling factor.
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
            mean_abs_w = torch.mean(torch.abs(self.weight))
            beta = torch.clamp(mean_abs_w, min=1e-5)
            
            w_scaled = self.weight / beta
            w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
            w_final = self.weight + (w_quant * beta - self.weight).detach()

            max_x = torch.max(torch.abs(x)) + 1e-5
            x_scaled = x * (127.0 / max_x)
            x_quant = torch.clamp(torch.round(x_scaled), -128.0, 127.0)
            x_final = x + (x_quant * (max_x / 127.0) - x).detach()

            return F.linear(x_final, w_final, self.bias)


    class BitAttention(nn.Module):
        """
        Causal Self-Attention Layer using 1-Bit (BitLinear) Projections.
        """
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
        """
        Multi-Layer Perceptron using 1-Bit layers.
        """
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
            x = x + self.attn(self.ln1(x), mask)
            x = x + self.mlp(self.ln2(x))
            return x


    class BitTransformerLM(nn.Module):
        """
        Full 1-Bit Autoregressive Language Model.
        """
        def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, seq_len: int):
            super().__init__()
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.position_embedding = nn.Embedding(seq_len, embed_dim)
            
            self.blocks = nn.ModuleList([
                BitTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
            ])
            
            self.ln_f = nn.LayerNorm(embed_dim)
            self.lm_head = BitLinear(embed_dim, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
            batch_size, seq_len = input_ids.size()
            device = input_ids.device
            
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len)
            
            for block in self.blocks:
                x = block(x, mask)
                
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
                
            return logits, loss

        def generate(self, prompt_ids: List[int], max_new_tokens: int, temperature: float = 1.0) -> List[int]:
            self.eval()
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            
            for _ in range(max_new_tokens):
                context_ids = input_ids[:, -self.seq_len:]
                with torch.no_grad():
                    logits, _ = self.forward(context_ids)
                    
                next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                
                if next_token.item() == 2:  # EOS token
                    break
                    
            return input_ids[0].tolist()


# =====================================================================
# PURE NUMPY 1-BIT TRANSFORMR DEPLOYMENT (For PyTorch-exported weights)
# =====================================================================

class NumPyBitLinear:
    def __init__(self, weight: np.ndarray, bias: np.ndarray = None):
        self.weight = weight
        self.bias = bias
        
        mean_abs_w = np.mean(np.abs(self.weight))
        self.beta = max(mean_abs_w, 1e-5)
        self.w_quant = np.clip(np.round(self.weight / self.beta), -1.0, 1.0) * self.beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        max_x = np.max(np.abs(x)) + 1e-5
        x_quant = np.clip(np.round(x * (127.0 / max_x)), -128.0, 127.0) * (max_x / 127.0)
        
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
            scores = np.where(mask == 0, -1e9, scores)
            
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
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
        
        self.ln1_w = block_data["ln1_w"]
        self.ln1_b = block_data["ln1_b"]
        self.ln2_w = block_data["ln2_w"]
        self.ln2_b = block_data["ln2_b"]
        
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
        norm_x1 = self.layernorm(x, self.ln1_w, self.ln1_b)
        x = x + self.attn.forward(norm_x1, mask)
        norm_x2 = self.layernorm(x, self.ln2_w, self.ln2_b)
        x = x + self.mlp.forward(norm_x2)
        return x


class NumPyBitTransformerLM:
    """
    Pure NumPy transformer engine. Runs 1-bit Transformer models on mobile.
    """
    def __init__(self, weights_dict: dict):
        self.token_emb = weights_dict["token_embedding"]
        self.pos_emb = weights_dict["position_embedding"]
        
        self.vocab_size = self.token_emb.shape[0]
        self.embed_dim = self.token_emb.shape[1]
        self.seq_len = self.pos_emb.shape[0]
        self.num_heads = weights_dict["num_heads"]
        
        self.ln_f_w = weights_dict["ln_f_w"]
        self.ln_f_b = weights_dict["ln_f_b"]
        self.lm_head = NumPyBitLinear(weights_dict["lm_head_w"])
        
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
            
        x = self.layernorm(x, self.ln_f_w, self.ln_f_b)
        logits = self.lm_head.forward(x)
        return logits

    def generate(self, prompt_ids: List[int], max_new_tokens: int, temperature: float = 1.0) -> List[int]:
        input_ids = np.array([prompt_ids], dtype=np.int32)
        
        for _ in range(max_new_tokens):
            context_ids = input_ids[:, -self.seq_len:]
            logits = self.forward(context_ids)
            next_logits = logits[0, -1, :] / max(temperature, 1e-5)
            exp_logits = np.exp(next_logits - np.max(next_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_token = np.random.choice(self.vocab_size, p=probs)
            input_ids = np.concatenate([input_ids, np.array([[next_token]])], axis=1)
            
            if next_token == 2:
                break
                
        return input_ids[0].tolist()


# =====================================================================
# PURE NUMPY 1-BIT RNN RECURRENT MODEL (For pure NumPy training fallback)
# =====================================================================

class NumPyBitRNNLM:
    """
    Pure NumPy 1-Bit Recurrent Language Model with BPTT and STE.
    Runs and trains perfectly without PyTorch, ideal for mobile ARM64.
    """
    def __init__(self, vocab_size: int, embed_dim: int, seq_len: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.hidden_dim = embed_dim
        
        # Continuous weights (initialized small)
        self.E = np.random.randn(vocab_size, embed_dim) * 0.1
        self.W_xh = np.random.randn(self.hidden_dim, embed_dim) * 0.1
        self.W_hh = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        self.W_hy = np.random.randn(vocab_size, self.hidden_dim) * 0.1
        
        self.b_h = np.zeros((self.hidden_dim, 1))
        self.b_y = np.zeros((vocab_size, 1))
        
        # Adam Optimizer states
        self.m_E, self.v_E = np.zeros_like(self.E), np.zeros_like(self.E)
        self.m_W_xh, self.v_W_xh = np.zeros_like(self.W_xh), np.zeros_like(self.W_xh)
        self.m_W_hh, self.v_W_hh = np.zeros_like(self.W_hh), np.zeros_like(self.W_hh)
        self.m_W_hy, self.v_W_hy = np.zeros_like(self.W_hy), np.zeros_like(self.W_hy)
        self.m_b_h, self.v_b_h = np.zeros_like(self.b_h), np.zeros_like(self.b_h)
        self.m_b_y, self.v_b_y = np.zeros_like(self.b_y), np.zeros_like(self.b_y)
        
        self.t = 0

    def q_weights(self, W: np.ndarray):
        """Quantizes weights to ternary values {-1, 0, 1} with scale factor."""
        beta = np.mean(np.abs(W)) + 1e-5
        W_q = np.clip(np.round(W / beta), -1.0, 1.0)
        return W_q * beta, W_q, beta

    def q_activation(self, x: np.ndarray) -> np.ndarray:
        """Quantizes activation to 8-bit."""
        max_x = np.max(np.abs(x)) + 1e-5
        return np.clip(np.round(x * (127.0 / max_x)), -128.0, 127.0) * (max_x / 127.0)

    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, lr: float = 0.005) -> float:
        """
        Runs one step of BPTT training with Straight-Through Estimator.
        Updates model weights using Adam optimizer.
        """
        # Batch sizes and seq lengths
        batch_size, seq_len = input_ids.shape
        
        # Shape: (seq_len, batch_size)
        X = input_ids.T
        Y = target_ids.T
        
        # 1-bit quantized forward weights
        W_xh_f, W_xh_q, b_xh = self.q_weights(self.W_xh)
        W_hh_f, W_hh_q, b_hh = self.q_weights(self.W_hh)
        W_hy_f, W_hy_q, b_hy = self.q_weights(self.W_hy)
        
        # Cached activations for backprop
        h = {}
        h[-1] = np.zeros((self.hidden_dim, batch_size))
        
        x_emb = {}
        a = {}
        probs = {}
        loss = 0.0
        
        # 1. Forward Pass
        for t in range(seq_len):
            # Token embedding
            x_emb[t] = self.q_activation(self.E[X[t]].T)  # (hidden_dim, batch_size)
            
            # Recurrent hidden transition
            a[t] = np.dot(W_xh_f, x_emb[t]) + np.dot(W_hh_f, h[t-1]) + self.b_h
            h[t] = np.tanh(a[t])
            
            # Output logits
            logits_t = np.dot(W_hy_f, h[t]) + self.b_y
            
            # Softmax
            exp_logits = np.exp(logits_t - np.max(logits_t, axis=0, keepdims=True))
            probs[t] = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
            
            # Cross entropy loss calculation
            targets_idx = Y[t]
            loss += -np.log(probs[t][targets_idx, np.arange(batch_size)] + 1e-15).mean()
            
        loss /= seq_len
        
        # 2. Backward Pass (BPTT with STE)
        dE = np.zeros_like(self.E)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros((self.hidden_dim, batch_size))
        
        for t in reversed(range(seq_len)):
            # Target output gradient
            dy = probs[t].copy()
            dy[Y[t], np.arange(batch_size)] -= 1.0
            dy /= (batch_size * seq_len)  # Normalize
            
            # LM head output projections gradients
            dW_hy += np.dot(dy, h[t].T)
            db_y += np.sum(dy, axis=1, keepdims=True)
            
            # Hidden gradient
            dh = np.dot(W_hy_f.T, dy) + dh_next
            
            # Backprop through tanh activation
            da = dh * (1.0 - h[t]**2)
            
            # Linear projection gradients
            dW_xh += np.dot(da, x_emb[t].T)
            dW_hh += np.dot(da, h[t-1].T)
            db_h += np.sum(da, axis=1, keepdims=True)
            
            # Backprop to token embedding
            dx = np.dot(W_xh_f.T, da)
            for b in range(batch_size):
                dE[X[t, b]] += dx[:, b]
                
            # Transition to previous hidden step
            dh_next = np.dot(W_hh_f.T, da)
            
        # 3. Adam Optimizer Update Steps
        self.t += 1
        eps = 1e-8
        beta1, beta2 = 0.9, 0.999
        
        # Clip gradients to avoid exploding/vanishing gradients in deep BPTT
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
        
        return loss

    def generate(self, prompt_ids: List[int], max_new_tokens: int, temperature: float = 1.0) -> List[int]:
        """Autoregressive text generation using trained Recurrent parameters."""
        input_ids = list(prompt_ids)
        
        # Initialize hidden state
        h_prev = np.zeros((self.hidden_dim, 1))
        
        # Pre-fill hidden states with prompt tokens
        W_xh_f, _, _ = self.q_weights(self.W_xh)
        W_hh_f, _, _ = self.q_weights(self.W_hh)
        W_hy_f, _, _ = self.q_weights(self.W_hy)
        
        for p_id in input_ids:
            x_t = self.q_activation(self.E[p_id].reshape(-1, 1))
            h_prev = np.tanh(np.dot(W_xh_f, x_t) + np.dot(W_hh_f, h_prev) + self.b_h)
            
        for _ in range(max_new_tokens):
            x_t = self.q_activation(self.E[input_ids[-1]].reshape(-1, 1))
            h_prev = np.tanh(np.dot(W_xh_f, x_t) + np.dot(W_hh_f, h_prev) + self.b_h)
            
            logits = np.dot(W_hy_f, h_prev) + self.b_y
            logits = logits.flatten() / max(temperature, 1e-5)
            
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_token = np.random.choice(self.vocab_size, p=probs)
            input_ids.append(int(next_token))
            
            if next_token == 2: # EOS
                break
                
        return input_ids
