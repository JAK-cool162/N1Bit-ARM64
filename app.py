import os
import sys
import json
import time
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from typing import List, Dict

# Add parent directory to path to ensure correct package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from n1bit.config import (
    LINKS_FILE, CACHE_DIR, MODEL_PROFILES,
    EMBED_DIM, NUM_LAYERS, NUM_HEADS, SEQ_LEN, get_model_paths, USE_16BIT
)
from n1bit.tokenizer import SimpleBPETokenizer
from n1bit.trainer import Trainer
from n1bit.dataset import DatasetEngine
from n1bit.model import NumPyBitRNNLM, NumPyBitTransformerLM

try:
    import torch
    from n1bit.model import BitTransformerLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

app = Flask(__name__)

# Global state to keep track of active training sessions, active models, etc.
global_state = {
    "active_model": "default",
    "training_thread": None,
    "training_logs": [],
    "is_training": False,
    "current_step": 0,
    "current_epoch": 0,
    "current_loss": 0.0,
    "current_speed": 0.0,
    "training_progress_pct": 0,
}

def load_model_dimensions_app(model_name: str) -> tuple:
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
        except Exception:
            pass
            
    return e_dim, n_layers, n_heads, s_len

# =====================================================================
# CORE API ENDPOINTS
# =====================================================================

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns the current status of the AI engine and available models."""
    models = ["default"]
    if os.path.exists(CACHE_DIR):
        for item in os.listdir(CACHE_DIR):
            if os.path.isdir(os.path.join(CACHE_DIR, item)):
                if any(f in os.listdir(os.path.join(CACHE_DIR, item)) for f in ["tokenizer.json", "numpy_weights.npz", "model_checkpoint.pt"]):
                    if item not in models:
                        models.append(item)
                        
    paths = get_model_paths(global_state["active_model"])
    model_loaded = os.path.exists(paths["numpy_weights"]) or os.path.exists(paths["checkpoint"])
    
    # Load profile details
    embed_dim, num_layers, num_heads, seq_len = load_model_dimensions_app(global_state["active_model"])
    
    return jsonify({
        "active_model": global_state["active_model"],
        "is_training": global_state["is_training"],
        "has_torch": HAS_TORCH,
        "models": models,
        "model_loaded": model_loaded,
        "dimensions": {
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "seq_len": seq_len
        },
        "stats": {
            "epoch": global_state["current_epoch"],
            "step": global_state["current_step"],
            "loss": round(global_state["current_loss"], 4),
            "speed": round(global_state["current_speed"], 1)
        }
    })

@app.route("/api/select_model", methods=["POST"])
def select_model():
    """Changes the active named model."""
    data = request.json or {}
    model_name = data.get("model_name", "default").strip()
    if not model_name:
        return jsonify({"error": "Model name cannot be empty"}), 400
        
    global_state["active_model"] = model_name
    return jsonify({"success": True, "active_model": model_name})

@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    """Lists all dataset links in links.txt and pre-processing statistics."""
    temp_engine = DatasetEngine("cache/temp.jsonl", "cache/temp.json")
    urls = temp_engine.read_links()
    repos = [temp_engine.parse_repo_id(url) for url in urls]
    
    paths = get_model_paths(global_state["active_model"])
    stats_data = {}
    if os.path.exists(paths["stats"]):
        try:
            with open(paths["stats"], "r") as f:
                stats_data = json.load(f)
        except Exception:
            pass
            
    return jsonify({
        "urls": urls,
        "repos": repos,
        "stats": stats_data
    })

@app.route("/api/add_dataset", methods=["POST"])
def add_dataset():
    """Adds a new dataset URL to links.txt."""
    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400
        
    existing = []
    if os.path.exists(LINKS_FILE):
        with open(LINKS_FILE, 'r', encoding='utf-8') as f:
            existing = [line.strip() for line in f if line.strip()]
            
    if url in existing:
        return jsonify({"success": True, "info": "URL already exists in links.txt"})
        
    with open(LINKS_FILE, 'a', encoding='utf-8') as f:
        f.write(url + "\n")
        
    return jsonify({"success": True, "url": url})

@app.route("/api/neuron_states", methods=["GET"])
def get_neuron_states():
    """Analyzes the current active weights of the model to show neuron grid states (+1 vs -1)."""
    paths = get_model_paths(global_state["active_model"])
    
    weights_file = paths["numpy_weights"]
    if not os.path.exists(weights_file):
        return jsonify({"error": "No trained model weights found for this model yet."}), 404
        
    try:
        data = np.load(weights_file, allow_pickle=True)
        model_type = str(data.get("model_type", "transformer"))
        
        flat_signs = []
        if model_type == "rnn":
            flat_signs = np.sign(data["W_xh"]).flatten()
        else:
            flat_signs = np.sign(data["block_0_q_proj_w"]).flatten()
            
        total = len(flat_signs)
        positives = int(np.sum(flat_signs >= 0))
        negatives = int(np.sum(flat_signs < 0))
        
        sample_indices = np.random.choice(total, min(total, 64), replace=False)
        sample_grid = [1 if flat_signs[i] >= 0 else -1 for i in sample_indices]
        
        return jsonify({
            "model_name": global_state["active_model"],
            "model_type": model_type,
            "total_neurons": total,
            "positives": positives,
            "negatives": negatives,
            "positive_pct": round((positives / total) * 100, 2),
            "negative_pct": round((negatives / total) * 100, 2),
            "grid": sample_grid
        })
    except Exception as e:
        return jsonify({"error": f"Failed to analyze weights: {e}"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Runs causal autoregressive chat generation and returns predictions & alternative probabilities."""
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 40))
    
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400
        
    paths = get_model_paths(global_state["active_model"])
    
    if not os.path.exists(paths["tokenizer"]):
        return jsonify({"error": "No tokenizer found. Please train the model first."}), 400
        
    tokenizer = SimpleBPETokenizer()
    tokenizer.load(paths["tokenizer"])
    
    # Load dimensions
    embed_dim, num_layers, num_heads, seq_len = load_model_dimensions_app(global_state["active_model"])
    
    formatted_prompt = f"<instruction>: {prompt}\n<response>:"
    prompt_ids = tokenizer.encode(formatted_prompt)
    prompt_ids = prompt_ids[-seq_len+20:]
    prompt_ids = [tokenizer.bos_id] + prompt_ids
    
    if not os.path.exists(paths["numpy_weights"]):
        return jsonify({"error": "No trained weights found. Please train the model first."}), 400
        
    try:
        npz_data = np.load(paths["numpy_weights"], allow_pickle=True)
        model_type = str(npz_data.get("model_type", "transformer"))
        
        if model_type == "rnn":
            vocab_size = int(npz_data["vocab_size"])
            
            model = NumPyBitRNNLM(vocab_size=vocab_size, embed_dim=embed_dim, seq_len=seq_len)
            model.E = npz_data["E"]
            model.W_xh = npz_data["W_xh"]
            model.W_hh = npz_data["W_hh"]
            model.W_hy = npz_data["W_hy"]
            model.b_h = npz_data["b_h"]
            model.b_y = npz_data["b_y"]
            
            h_prev = np.zeros((model.hidden_dim, 1), dtype=np.float16)
            W_xh_f, _, _ = model.q_weights(model.W_xh)
            W_hh_f, _, _ = model.q_weights(model.W_hh)
            W_hy_f, _, _ = model.q_weights(model.W_hy)
            
            for p_id in prompt_ids:
                x_t = model.q_activation(model.E[p_id].reshape(-1, 1))
                h_prev = np.tanh(np.dot(W_xh_f, x_t) + np.dot(W_hh_f, h_prev) + model.b_h)
                
            logits = np.dot(W_hy_f, h_prev) + model.b_y
            next_logits = logits.flatten().astype(np.float32)
        else:
            weights_dict = {
                "num_heads": num_heads,
                "token_embedding": npz_data["token_embedding"],
                "position_embedding": npz_data["position_embedding"],
                "ln_f_w": npz_data["ln_f_w"],
                "ln_f_b": npz_data["ln_f_b"],
                "lm_head_w": npz_data["lm_head_w"],
                "blocks": []
            }
            
            block_keys = [k for k in npz_data.keys() if k.startswith("block_")]
            num_blocks = len(set(int(k.split("_")[1]) for k in block_keys))
            for i in range(num_blocks):
                block_data = {
                    "ln1_w": npz_data[f"block_{i}_ln1_w"],
                    "ln1_b": npz_data[f"block_{i}_ln1_b"],
                    "ln2_w": npz_data[f"block_{i}_ln2_w"],
                    "ln2_b": npz_data[f"block_{i}_ln2_b"],
                    "q_proj_w": npz_data[f"block_{i}_q_proj_w"],
                    "k_proj_w": npz_data[f"block_{i}_k_proj_w"],
                    "v_proj_w": npz_data[f"block_{i}_v_proj_w"],
                    "out_proj_w": npz_data[f"block_{i}_out_proj_w"],
                    "gate_proj_w": npz_data[f"block_{i}_gate_proj_w"],
                    "down_proj_w": npz_data[f"block_{i}_down_proj_w"]
                }
                weights_dict["blocks"].append(block_data)
                
            model = NumPyBitTransformerLM(weights_dict)
            context_ids = np.array([prompt_ids], dtype=np.int32)
            logits = model.forward(context_ids)
            next_logits = logits[0, -1, :].astype(np.float32)

        exp_logits = np.exp(next_logits - np.max(next_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top_indices = np.argsort(probs)[::-1][:5]
        alternatives = []
        for rank, idx in enumerate(top_indices, 1):
            token_str = tokenizer.inverse_vocab.get(str(idx), tokenizer.inverse_vocab.get(idx, "<unk>"))
            alternatives.append({
                "rank": rank,
                "token": token_str,
                "probability": float(probs[idx])
            })
            
        output_ids = model.generate(prompt_ids, max_new_tokens=max_tokens, temperature=temperature)
        new_ids = output_ids[len(prompt_ids):]
        response_text = tokenizer.decode(new_ids).strip()
        
        if not response_text:
            response_text = "[N1Bit Engine is thinking...] (Please train the model longer on your chosen datasets to get complete speech patterns)"
            
        return jsonify({
            "response": response_text,
            "alternatives": alternatives
        })
    except Exception as e:
        return jsonify({"error": f"Failed during inference: {e}"}), 500

# =====================================================================
# REAL-TIME TRAINING PIPELINE THREAD
# =====================================================================

def run_training_worker(model_name: str, limit_steps: str, selected_repos: List[str]):
    """Background worker thread to run pre-training and log metrics."""
    global_state["is_training"] = True
    global_state["training_logs"] = ["[System] Initializing training thread..."]
    global_state["current_step"] = 0
    global_state["current_epoch"] = 1
    global_state["current_loss"] = 0.0
    global_state["current_speed"] = 0.0
    
    try:
        trainer = Trainer(model_name=model_name, limit_steps=limit_steps)
        
        # Load dimensions
        embed_dim, num_layers, num_heads, seq_len = load_model_dimensions_app(model_name)
        
        # Inject custom dimensions into n1bit trainer package globally
        import n1bit.trainer as t_mod
        t_mod.EMBED_DIM = embed_dim
        t_mod.NUM_LAYERS = num_layers
        t_mod.NUM_HEADS = num_heads
        t_mod.SEQ_LEN = seq_len
        
        def log_cb(msg):
            global_state["training_logs"].append(msg)
            if len(global_state["training_logs"]) > 200:
                global_state["training_logs"].pop(0)
                
        trainer.engine.process_all_datasets(selected_repos=selected_repos)
        trainer.prepare_tokenizer()
        
        vocab_size = len(trainer.tokenizer.vocab)
        start_epoch = 1
        start_step = 0
        loss_history = []
        
        resumed = False
        progress_path = trainer.paths["progress"]
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)
                start_epoch = progress_data.get("epoch", 1)
                start_step = progress_data.get("step", 0)
                loss_history = progress_data.get("loss_history", [])
                log_cb(f"[System] Resuming training from Epoch {start_epoch}, Step {start_step}...")
                resumed = True
            except Exception:
                pass

        is_infinite = (trainer.limit_steps == float('inf'))
        target_epochs = 100000 if is_infinite else trainer.limit_steps if trainer.limit_steps else 1
        
        checkpoint_path = trainer.paths["checkpoint"]
        numpy_weight_path = trainer.paths["numpy_weights"]
        
        log_cb(f"[System] Initializing model '{model_name}' (16-bit Q1 Recurrent RNN mode)...")
        model = NumPyBitRNNLM(vocab_size=vocab_size, embed_dim=embed_dim, seq_len=seq_len)
        
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
                log_cb("[System] Checkpoint weights and Adam states loaded successfully.")
            except Exception as e:
                log_cb(f"[Warning] Failed to load checkpoint: {e}. Starting fresh.")
                
        log_cb("[System] Training loop started!")
        step = 0
        total_loss = 0.0
        start_time = time.time()
        
        epoch = start_epoch
        while global_state["is_training"]:
            chunk_gen = trainer.get_token_chunk_stream()
            batch_gen = trainer.get_batch_generator(chunk_gen, BATCH_SIZE)
            
            for batch_data in batch_gen:
                if not global_state["is_training"]:
                    break
                    
                step += 1
                
                if epoch == start_epoch and step <= start_step:
                    continue
                    
                batch_np = np.array(batch_data, dtype=np.int32)
                x = batch_np[:, :-1]
                y = batch_np[:, 1:]
                
                loss = model.train_step(x, y, lr=LR)
                total_loss += loss
                
                global_state["current_step"] = step
                global_state["current_epoch"] = epoch
                global_state["current_loss"] = float(loss)
                
                if step % 5 == 0 or step == 1:
                    avg_loss = total_loss / (1.0 if step == 1 or total_loss == loss else 5.0)
                    loss_history.append({"epoch": epoch, "step": step, "loss": avg_loss})
                    
                    elapsed = time.time() - start_time
                    tokens_per_sec = (step * BATCH_SIZE * seq_len) / max(elapsed, 1e-5)
                    global_state["current_speed"] = tokens_per_sec
                    
                    log_msg = f"Epoch {epoch} | Step {step:4d} | Loss: {avg_loss:.4f} | Speed: {tokens_per_sec:.1f} tok/sec"
                    log_cb(log_msg)
                    total_loss = 0.0
                    
                    trainer.save_progress(epoch, step, loss_history)
                    np.savez_compressed(
                        numpy_weight_path,
                        model_type=np.array("rnn"),
                        vocab_size=np.array(vocab_size),
                        embed_dim=np.array(embed_dim),
                        seq_len=np.array(seq_len),
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
                    
                if not is_infinite and trainer.limit_steps and step >= trainer.limit_steps:
                    log_cb(f"[System] Training successfully finished. Reached step limit: {trainer.limit_steps}.")
                    break
                    
            if not global_state["is_training"]:
                break
                
            if not is_infinite and trainer.limit_steps and step >= trainer.limit_steps:
                break
                
            epoch += 1
            if not is_infinite and epoch > target_epochs:
                log_cb("[System] Training successfully finished. Reached epoch limits.")
                break
            start_step = 0
            
        np.savez_compressed(
            numpy_weight_path,
            model_type=np.array("rnn"),
            vocab_size=np.array(vocab_size),
            embed_dim=np.array(embed_dim),
            seq_len=np.array(seq_len),
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
        log_cb("[System] Final weights saved and synced successfully.")
        
    except Exception as e:
        global_state["training_logs"].append(f"[Error] Training crashed: {e}")
    finally:
        global_state["is_training"] = False

@app.route("/api/start_train", methods=["POST"])
def start_train():
    """Starts the model pre-training in a background thread."""
    if global_state["is_training"]:
        return jsonify({"error": "Training is already running"}), 400
        
    data = request.json or {}
    steps = data.get("steps", "inf").strip()
    selected_repos = data.get("selected_repos", None)
    
    global_state["training_thread"] = threading.Thread(
        target=run_training_worker,
        args=(global_state["active_model"], steps, selected_repos)
    )
    global_state["training_thread"].daemon = True
    global_state["training_thread"].start()
    
    return jsonify({"success": True, "message": "Training started"})

@app.route("/api/stop_train", methods=["POST"])
def stop_train():
    """Stops the active pre-training session gracefully."""
    if not global_state["is_training"]:
        return jsonify({"error": "Training is not running"}), 400
        
    global_state["is_training"] = False
    return jsonify({"success": True, "message": "Stopping training..."})

@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns the background training console logs and latest metrics."""
    return jsonify({
        "logs": global_state["training_logs"],
        "is_training": global_state["is_training"],
        "step": global_state["current_step"],
        "epoch": global_state["current_epoch"],
        "loss": round(global_state["current_loss"], 4),
        "speed": round(global_state["current_speed"], 1)
    })

@app.route("/api/create_named_model", methods=["POST"])
def create_named_model():
    """Creates a new named model and saves its selected hardware profile dimensions."""
    data = request.json or {}
    name = data.get("name", "").strip()
    profile_id = data.get("profile", "2").strip()
    
    if not name:
        return jsonify({"error": "Model name cannot be empty"}), 400
        
    profile = MODEL_PROFILES.get(profile_id, MODEL_PROFILES["2"])
    paths = get_model_paths(name)
    config_path = os.path.join(paths["model_dir"], "model_config.json")
    
    # Calculate parameter count (assuming vocab_size around 4000)
    vocab_size = 4000
    estimated_params = calculate_parameter_count(
        vocab_size, profile["embed_dim"], profile["num_layers"], profile["seq_len"]
    )
    
    model_config = {
        "model_name": name,
        "embed_dim": profile["embed_dim"],
        "num_layers": profile["num_layers"],
        "num_heads": profile["num_heads"],
        "seq_len": profile["seq_len"],
        "estimated_params": estimated_params
    }
    
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)
            
        global_state["active_model"] = name
        return jsonify({"success": True, "active_model": name, "config": model_config})
    except Exception as e:
        return jsonify({"error": f"Failed to save model config: {e}"}), 500

# =====================================================================
# HTML EMBEDDED DASHBOARD PAGE (Tailwind CSS, Touch-optimized Mobile App UI)
# =====================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>N1Bit-ARM64 | 1-Bit AI Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#f5f3ff',
                            100: '#ede9fe',
                            200: '#ddd6fe',
                            500: '#8b5cf6',
                            600: '#7c3aed',
                            700: '#6d28d9',
                            800: '#5b21b6',
                            900: '#4c1d95',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Inter', sans-serif; -webkit-tap-highlight-color: transparent; }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(139, 92, 246, 0.3); border-radius: 4px; }
    </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen flex flex-col custom-scrollbar">

    <!-- HEADER -->
    <header class="border-b border-gray-800 bg-gray-900/60 backdrop-blur-md sticky top-0 z-50 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-between text-white font-bold text-center justify-center text-lg shadow-lg shadow-brand-500/20">
                1
            </div>
            <div>
                <h1 class="text-md font-bold tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">N1Bit-ARM64</h1>
                <p class="text-xs text-gray-500 font-medium">1-Bit AI Mobile Framework</p>
            </div>
        </div>
        
        <!-- Model Selector Dropdown -->
        <div class="flex items-center space-x-2">
            <select id="model-select" onchange="changeModel()" class="bg-gray-800 text-xs text-gray-200 border border-gray-700 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-500 font-semibold cursor-pointer">
                <option value="default">default</option>
            </select>
            <button onclick="openCreateModelModal()" class="p-1.5 bg-gray-800 text-gray-400 hover:text-white border border-gray-700 rounded-lg" title="Create Named Model">
                <i class="fa-solid fa-plus text-xs"></i>
            </button>
        </div>
    </header>

    <!-- NEW MODEL MODAL -->
    <div id="new-model-modal" class="hidden fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-gray-900 border border-gray-800 w-full max-w-md rounded-2xl p-5 space-y-4">
            <div class="flex justify-between items-center border-b border-gray-800 pb-2">
                <h3 class="text-sm font-bold text-white uppercase tracking-wider">Initialize Named Model</h3>
                <button onclick="closeCreateModelModal()" class="text-gray-500 hover:text-white"><i class="fa-solid fa-xmark text-sm"></i></button>
            </div>
            
            <div class="space-y-3">
                <div>
                    <label class="block text-xs font-bold text-gray-400 mb-1">Model Name</label>
                    <input type="text" id="modal-model-name" placeholder="e.g. coder, sandbox, physics" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-brand-500 text-gray-200">
                </div>
                
                <div>
                    <label class="block text-xs font-bold text-gray-400 mb-1">Target Hardware Profile</label>
                    <select id="modal-profile" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-brand-500 text-gray-200 cursor-pointer">
                        <option value="1">BUDGET (~10k-50k params) - Old/Budget Chips</option>
                        <option value="2" selected>MIDRANGE (~100k-500k params) - Standard Chips</option>
                        <option value="3">FLAGSHIP (~1M-5M params) - High-End/Flagship Chips</option>
                        <option value="4">DESKTOP (10M+ params) - PCs & CUDA/Vulkan GPUs</option>
                    </select>
                    <p class="text-[10px] text-gray-500 mt-1">Avoid cramping a 7B model. Choose a size optimized for your hardware to prevent memory freeze!</p>
                </div>
            </div>
            
            <button onclick="submitNewModel()" class="w-full py-2 bg-brand-600 hover:bg-brand-700 text-white font-bold rounded-xl text-xs">
                Create and Select
            </button>
        </div>
    </div>

    <!-- MAIN BODY -->
    <main class="flex-1 max-w-6xl w-full mx-auto p-3 flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
        
        <!-- LEFT SIDEBAR: DASHBOARD METRICS -->
        <section class="w-full md:w-80 space-y-4 flex flex-col">
            
            <!-- Quick Status Card -->
            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-4 shadow-xl">
                <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Engine Status</h2>
                <div class="space-y-3">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Environment:</span>
                        <span id="env-badge" class="text-xs font-bold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400">ARM64 / Phone</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Precision Mode:</span>
                        <span class="text-xs font-bold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400">FP16 / Q1 Binary</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Model Checkpoint:</span>
                        <span id="checkpoint-badge" class="text-xs font-bold px-2 py-0.5 rounded-full bg-red-500/10 text-red-400">None</span>
                    </div>
                </div>
            </div>

            <!-- Model Dimensions Card -->
            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-4 shadow-xl">
                <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Model Architecture Specs</h2>
                <div class="space-y-2 text-xs">
                    <div class="flex justify-between py-1 border-b border-gray-800/40">
                        <span class="text-gray-400">Embedding Dim:</span>
                        <span id="dim-embed" class="font-bold text-gray-200">128</span>
                    </div>
                    <div class="flex justify-between py-1 border-b border-gray-800/40">
                        <span class="text-gray-400">Layers Count:</span>
                        <span id="dim-layers" class="font-bold text-gray-200">2</span>
                    </div>
                    <div class="flex justify-between py-1 border-b border-gray-800/40">
                        <span class="text-gray-400">Attention Heads:</span>
                        <span id="dim-heads" class="font-bold text-gray-200">2</span>
                    </div>
                    <div class="flex justify-between py-1 border-b border-gray-800/40">
                        <span class="text-gray-400">Max Sequence context:</span>
                        <span id="dim-seq" class="font-bold text-gray-200">64 tokens</span>
                    </div>
                </div>
            </div>

            <!-- Live Training Metrics -->
            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-4 shadow-xl">
                <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Pre-Training Metrics</h2>
                <div class="grid grid-cols-2 gap-3">
                    <div class="bg-gray-950 p-3 rounded-xl border border-gray-800 text-center">
                        <p class="text-xs text-gray-500">Epoch</p>
                        <p id="metric-epoch" class="text-lg font-extrabold text-brand-500">0</p>
                    </div>
                    <div class="bg-gray-950 p-3 rounded-xl border border-gray-800 text-center">
                        <p class="text-xs text-gray-500">Step</p>
                        <p id="metric-step" class="text-lg font-extrabold text-indigo-400">0</p>
                    </div>
                    <div class="bg-gray-950 p-3 rounded-xl border border-gray-800 text-center">
                        <p class="text-xs text-gray-500">Loss</p>
                        <p id="metric-loss" class="text-lg font-extrabold text-rose-400">0.0000</p>
                    </div>
                    <div class="bg-gray-950 p-3 rounded-xl border border-gray-800 text-center">
                        <p class="text-xs text-gray-500">Speed</p>
                        <p id="metric-speed" class="text-sm font-extrabold text-amber-400 mt-1">0.0 t/s</p>
                    </div>
                </div>
            </div>
            
            <!-- Add Dataset Form Quick Box -->
            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-4 shadow-xl">
                <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Add Custom Link</h2>
                <div class="flex space-x-2">
                    <input type="text" id="new-dataset-url" placeholder="Paste HF Dataset URL..." class="flex-1 bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-brand-500 text-gray-200">
                    <button onclick="addDatasetLink()" class="px-3 bg-brand-600 hover:bg-brand-700 text-white rounded-xl text-xs font-semibold">
                        Add
                    </button>
                </div>
            </div>

        </section>

        <!-- RIGHT WORKSPACE: TABS AND PANELS -->
        <section class="flex-1 bg-gray-900 border border-gray-800 rounded-2xl flex flex-col shadow-xl overflow-hidden min-h-[450px]">
            
            <!-- Tabs Menu Bar -->
            <div class="flex border-b border-gray-800 bg-gray-950/40">
                <button onclick="switchTab('tab-chat')" id="tab-btn-chat" class="flex-1 py-3 text-sm font-bold text-center border-b-2 border-brand-500 text-white flex items-center justify-center space-x-2">
                    <i class="fa-solid fa-comments"></i>
                    <span>Causal Chat</span>
                </button>
                <button onclick="switchTab('tab-train')" id="tab-btn-train" class="flex-1 py-3 text-sm font-bold text-center border-b-2 border-transparent text-gray-400 hover:text-white flex items-center justify-center space-x-2">
                    <i class="fa-solid fa-dumbbell"></i>
                    <span>Trainer Dashboard</span>
                </button>
                <button onclick="switchTab('tab-datasets')" id="tab-btn-datasets" class="flex-1 py-3 text-sm font-bold text-center border-b-2 border-transparent text-gray-400 hover:text-white flex items-center justify-center space-x-2">
                    <i class="fa-solid fa-database"></i>
                    <span>Data Center</span>
                </button>
            </div>

            <!-- TAB 1: CAUSAL CHAT INTERFACE -->
            <div id="tab-chat" class="flex-1 flex flex-col md:flex-row overflow-hidden">
                
                <!-- Chat Feed -->
                <div class="flex-1 flex flex-col p-4 border-b md:border-b-0 md:border-r border-gray-800 overflow-hidden">
                    
                    <!-- Chat Bubbles -->
                    <div id="chat-messages" class="flex-1 overflow-y-auto space-y-4 pr-1 custom-scrollbar min-h-[250px]">
                        <div class="flex items-start space-x-3">
                            <div class="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-xs font-bold">AI</div>
                            <div class="bg-gray-800 rounded-2xl px-4 py-2.5 text-sm max-w-[85%]">
                                Hello! I am your 1-Bit AI model running natively in half-precision FP16. Ask me anything!
                            </div>
                        </div>
                    </div>
                    
                    <!-- Chat Input -->
                    <div class="mt-4 flex space-x-2">
                        <input type="text" id="chat-input" onkeydown="handleChatKey(event)" placeholder="Type a message..." class="flex-1 bg-gray-950 border border-gray-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-brand-500 text-gray-200">
                        <button onclick="sendChatMessage()" class="w-12 bg-brand-600 hover:bg-brand-700 text-white rounded-xl flex items-center justify-center shadow-lg shadow-brand-500/20">
                            <i class="fa-solid fa-paper-plane text-sm"></i>
                        </button>
                    </div>
                </div>

                <!-- Live Thoughts & Neuron Grid Side Panel -->
                <div class="w-full md:w-64 p-4 space-y-4 bg-gray-950/20 overflow-y-auto custom-scrollbar">
                    
                    <!-- Top 5 Alternatives (What the AI is thinking) -->
                    <div class="space-y-2">
                        <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center space-x-1.5">
                            <i class="fa-solid fa-brain text-brand-500"></i>
                            <span>Alternative Thoughts</span>
                        </h3>
                        <div id="alternatives-box" class="space-y-2">
                            <p class="text-xs text-gray-500 italic">Send a message to see next-word probabilities...</p>
                        </div>
                    </div>
                    
                    <hr class="border-gray-800">

                    <!-- Visual Neuron Grid -->
                    <div class="space-y-2">
                        <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center space-x-1.5">
                            <i class="fa-solid fa-microchip text-indigo-400"></i>
                            <span>Active 1-Bit States</span>
                        </h3>
                        <div id="neuron-grid" class="grid grid-cols-8 gap-1.5 bg-gray-950 p-2 rounded-xl border border-gray-800/80">
                            <!-- JS will inject 64 neuron dots here -->
                        </div>
                        <p class="text-[10px] text-gray-500 text-center mt-1">Glow = +1 state | Dim = -1 state</p>
                    </div>

                </div>
            </div>

            <!-- TAB 2: TRAINER DASHBOARD -->
            <div id="tab-train" class="hidden flex-1 p-4 flex flex-col space-y-4 overflow-hidden">
                <div class="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
                    
                    <!-- Train Settings Form -->
                    <div class="flex-1 bg-gray-950/30 p-4 rounded-xl border border-gray-800 space-y-3">
                        <h3 class="text-sm font-semibold text-gray-200">Pre-Training Settings</h3>
                        <div>
                            <label class="block text-xs font-bold text-gray-400 mb-1">Max Steps</label>
                            <input type="text" id="train-steps" value="inf" class="w-full bg-gray-950 border border-gray-800 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-brand-500">
                            <p class="text-[10px] text-gray-500 mt-1">Enter a step number or 'inf' for continuous infinite training.</p>
                        </div>
                        
                        <!-- Dataset Selection Checklist -->
                        <div>
                            <label class="block text-xs font-bold text-gray-400 mb-1">Select Datasets to Train On</label>
                            <div id="dataset-checklist" class="max-h-[120px] overflow-y-auto border border-gray-800 rounded-lg p-2 bg-gray-950 space-y-1.5 custom-scrollbar text-xs">
                                <!-- JS will inject checkboxes here -->
                            </div>
                        </div>

                        <!-- Action Buttons -->
                        <div class="flex space-x-3 pt-2">
                            <button id="btn-start-train" onclick="startTraining()" class="flex-1 py-2.5 bg-brand-600 hover:bg-brand-700 text-white font-bold rounded-lg text-xs shadow-lg shadow-brand-500/25">
                                Start Pre-Training
                            </button>
                            <button id="btn-stop-train" onclick="stopTraining()" class="flex-1 py-2.5 bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg text-xs disabled:opacity-40" disabled>
                                Stop Training
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Console Console Logs -->
                <div class="flex-1 flex flex-col bg-gray-950 rounded-xl border border-gray-800 overflow-hidden min-h-[180px]">
                    <div class="px-4 py-2 border-b border-gray-800 bg-gray-900/40 flex items-center justify-between text-xs text-gray-400 font-bold">
                        <span>Console Logs</span>
                        <span id="train-status-text" class="text-brand-400 flex items-center space-x-1">
                            <span class="w-2 h-2 rounded-full bg-gray-500 animate-pulse"></span>
                            <span>Idle</span>
                        </span>
                    </div>
                    <pre id="console-logs" class="flex-1 p-3 text-[11px] font-mono text-gray-300 overflow-y-auto custom-scrollbar whitespace-pre-wrap select-text leading-relaxed">
[System] Waiting for training instructions...
                    </pre>
                </div>
            </div>

            <!-- TAB 3: DATA CENTER -->
            <div id="tab-datasets" class="hidden flex-1 p-4 overflow-y-auto custom-scrollbar space-y-4">
                
                <!-- Dataset Statistics Dashboard -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div class="bg-gray-950 p-4 rounded-xl border border-gray-800">
                        <p class="text-xs text-gray-500">Processed Corpora</p>
                        <p id="stats-datasets" class="text-2xl font-bold mt-1 text-brand-500">0</p>
                    </div>
                    <div class="bg-gray-950 p-4 rounded-xl border border-gray-800">
                        <p class="text-xs text-gray-500">Deduplicated Clean Size</p>
                        <p id="stats-size" class="text-2xl font-bold mt-1 text-indigo-400">0.00 MB</p>
                    </div>
                    <div class="bg-gray-950 p-4 rounded-xl border border-gray-800">
                        <p class="text-xs text-gray-500">Estimated Clean Tokens</p>
                        <p id="stats-tokens" class="text-2xl font-bold mt-1 text-emerald-400">0</p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    
                    <!-- Language Distribution and Duplication Card -->
                    <div class="bg-gray-950/40 p-4 rounded-xl border border-gray-800 space-y-3">
                        <h3 class="text-xs font-bold text-gray-300 uppercase tracking-wider">Metrics and Language Layout</h3>
                        <div class="space-y-2 text-xs">
                            <div class="flex justify-between items-center py-1 border-b border-gray-800/60">
                                <span class="text-gray-400">Fingerprint Duplicate Rate:</span>
                                <span id="stats-dup" class="font-bold text-rose-400">0.0%</span>
                            </div>
                            <div class="flex justify-between items-center py-1 border-b border-gray-800/60">
                                <span class="text-gray-400">Passed QC Samples Ratio:</span>
                                <span id="stats-qc" class="font-bold text-emerald-400">0.0%</span>
                            </div>
                        </div>
                        <div class="space-y-1.5 pt-1">
                            <p class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Language Distribution</p>
                            <div id="lang-bar-container" class="space-y-2 text-xs">
                                <!-- JS will inject progress bars for languages -->
                            </div>
                        </div>
                    </div>

                    <!-- Repos list -->
                    <div class="bg-gray-950/40 p-4 rounded-xl border border-gray-800 flex flex-col max-h-[300px]">
                        <h3 class="text-xs font-bold text-gray-300 uppercase tracking-wider mb-2">Available Links from links.txt</h3>
                        <div id="raw-links-list" class="flex-1 overflow-y-auto space-y-1 text-xs pr-1 custom-scrollbar select-text">
                            <!-- JS will inject list here -->
                        </div>
                    </div>

                </div>
            </div>

        </section>
    </main>

    <!-- FOOTER -->
    <footer class="text-center py-3 border-t border-gray-800/80 text-[10px] text-gray-500 bg-gray-950">
        N1Bit-ARM64 AI Engine &bull; Built entirely in pure-Python & NumPy for low-power mobile deployment &bull; 2026
    </footer>

    <!-- JAVASCRIPT DASHBOARD LOGIC -->
    <script>
        let currentTab = "tab-chat";
        let activeModel = "default";
        let isTraining = false;
        let logInterval = null;

        // On document load
        window.addEventListener('load', () => {
            fetchStatus();
            fetchDatasets();
            fetchNeuronStates();
            
            // Loop statuses
            setInterval(fetchStatus, 3000);
            setInterval(fetchNeuronStates, 8000);
        });

        // 1. Fetch system and models status
        function fetchStatus() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    activeModel = data.active_model;
                    isTraining = data.is_training;
                    
                    // Update env badge if CPU has torch
                    const envBadge = document.getElementById('env-badge');
                    if (data.has_torch) {
                        envBadge.innerText = "PC / Accelerators Available";
                        envBadge.className = "text-xs font-bold px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400";
                    } else {
                        envBadge.innerText = "ARM64 / Mobile Fallback";
                        envBadge.className = "text-xs font-bold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400";
                    }
                    
                    // Update checkpoint status
                    const cpBadge = document.getElementById('checkpoint-badge');
                    if (data.model_loaded) {
                        cpBadge.innerText = "1-Bit Weights Active";
                        cpBadge.className = "text-xs font-bold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400";
                    } else {
                        cpBadge.innerText = "Untrained / No Weights";
                        cpBadge.className = "text-xs font-bold px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400";
                    }
                    
                    // Update dimension specs dynamically!
                    document.getElementById('dim-embed').innerText = data.dimensions.embed_dim;
                    document.getElementById('dim-layers').innerText = data.dimensions.num_layers;
                    document.getElementById('dim-heads').innerText = data.dimensions.num_heads;
                    document.getElementById('dim-seq').innerText = data.dimensions.seq_len + " tokens";
                    
                    // Update top metrics
                    document.getElementById('metric-epoch').innerText = data.stats.epoch;
                    document.getElementById('metric-step').innerText = data.stats.step;
                    document.getElementById('metric-loss').innerText = data.stats.loss > 0 ? data.stats.loss : "0.0000";
                    document.getElementById('metric-speed').innerText = data.stats.speed + " tok/s";
                    
                    // Update trainer buttons state
                    document.getElementById('btn-start-train').disabled = isTraining;
                    document.getElementById('btn-stop-train').disabled = !isTraining;
                    
                    if (isTraining && !logInterval) {
                        startLogging();
                    } else if (!isTraining && logInterval) {
                        stopLogging();
                    }
                    
                    // Update dropdown lists
                    const select = document.getElementById('model-select');
                    const currentValue = select.value;
                    
                    select.innerHTML = '';
                    data.models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m;
                        opt.innerText = m;
                        select.appendChild(opt);
                    });
                    
                    if (data.models.includes(currentValue)) {
                        select.value = currentValue;
                    } else {
                        select.value = data.active_model;
                    }
                });
        }

        // Change active model
        function changeModel() {
            const modelName = document.getElementById('model-select').value;
            fetch('/api/select_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName })
            })
            .then(res => res.json())
            .then(data => {
                fetchStatus();
                fetchDatasets();
                fetchNeuronStates();
            });
        }

        // Create Named Model Modal Functions
        function openCreateModelModal() {
            document.getElementById('new-model-modal').className = "fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4";
        }
        
        function closeCreateModelModal() {
            document.getElementById('new-model-modal').className = "hidden";
        }
        
        function submitNewModel() {
            const name = document.getElementById('modal-model-name').value.trim();
            const profile = document.getElementById('modal-profile').value;
            
            if (!name) {
                alert("Model name cannot be empty.");
                return;
            }
            
            fetch('/api/create_named_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, profile: profile })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    alert(`Successfully created named model '${name}' with optimized architecture parameters!`);
                    closeCreateModelModal();
                    fetchStatus();
                    fetchDatasets();
                    fetchNeuronStates();
                }
            });
        }

        // 2. Tab switcher
        function switchTab(tabId) {
            document.getElementById('tab-chat').className = tabId === 'tab-chat' ? 'flex-1 flex flex-col md:flex-row overflow-hidden' : 'hidden';
            document.getElementById('tab-train').className = tabId === 'tab-train' ? 'flex-1 p-4 flex flex-col space-y-4 overflow-hidden' : 'hidden';
            document.getElementById('tab-datasets').className = tabId === 'tab-datasets' ? 'flex-1 p-4 overflow-y-auto custom-scrollbar space-y-4' : 'hidden';
            
            document.getElementById('tab-btn-chat').className = tabId === 'tab-chat' ? 'flex-1 py-3 text-sm font-bold text-center border-b-2 border-brand-500 text-white flex items-center justify-center space-x-2' : 'flex-1 py-3 text-sm font-bold text-center border-b-2 border-transparent text-gray-400 hover:text-white flex items-center justify-center space-x-2';
            document.getElementById('tab-btn-train').className = tabId === 'tab-train' ? 'flex-1 py-3 text-sm font-bold text-center border-b-2 border-brand-500 text-white flex items-center justify-center space-x-2' : 'flex-1 py-3 text-sm font-bold text-center border-b-2 border-transparent text-gray-400 hover:text-white flex items-center justify-center space-x-2';
            document.getElementById('tab-btn-datasets').className = tabId === 'tab-datasets' ? 'flex-1 py-3 text-sm font-bold text-center border-b-2 border-brand-500 text-white flex items-center justify-center space-x-2' : 'flex-1 py-3 text-sm font-bold text-center border-b-2 border-transparent text-gray-400 hover:text-white flex items-center justify-center space-x-2';
            
            currentTab = tabId;
        }

        // 3. Fetch list of datasets and statistics
        function fetchDatasets() {
            fetch('/api/datasets')
                .then(res => res.json())
                .then(data => {
                    const checklist = document.getElementById('dataset-checklist');
                    checklist.innerHTML = '';
                    
                    data.repos.forEach(repo => {
                        const div = document.createElement('div');
                        div.className = "flex items-center space-x-2";
                        div.innerHTML = `
                            <input type="checkbox" id="check-${repo}" value="${repo}" class="rounded text-brand-500 focus:ring-brand-500 cursor-pointer">
                            <label for="check-${repo}" class="cursor-pointer font-medium select-none truncate text-gray-300" title="${repo}">${repo}</label>
                        `;
                        checklist.appendChild(div);
                    });
                    
                    const linksList = document.getElementById('raw-links-list');
                    linksList.innerHTML = '';
                    data.repos.forEach(repo => {
                        const div = document.createElement('div');
                        div.className = "py-1.5 border-b border-gray-800/40 text-gray-400 truncate hover:text-gray-200 cursor-pointer";
                        div.innerHTML = `<i class="fa-solid fa-link text-[10px] mr-1.5 text-brand-400"></i>${repo}`;
                        linksList.appendChild(div);
                    });
                    
                    const stats = data.stats;
                    if (stats.num_datasets_processed) {
                        document.getElementById('stats-datasets').innerText = stats.num_datasets_processed;
                        document.getElementById('stats-size').innerText = (stats.size_after_bytes / (1024 * 1024)).toFixed(2) + " MB";
                        document.getElementById('stats-tokens').innerText = stats.estimated_token_count.toLocaleString();
                        document.getElementById('stats-dup').innerText = stats.duplicate_rate + "%";
                        
                        const kept = stats.num_samples_kept;
                        const discarded = stats.num_samples_discarded;
                        const ratio = kept / (kept + discarded) * 100;
                        document.getElementById('stats-qc').innerText = ratio.toFixed(1) + "%";
                        
                        const langContainer = document.getElementById('lang-bar-container');
                        langContainer.innerHTML = '';
                        
                        const totalLangs = Object.values(stats.language_distribution).reduce((a, b) => a + b, 0);
                        Object.entries(stats.language_distribution).forEach(([lang, count]) => {
                            const pct = (count / totalLangs * 100).toFixed(1);
                            const bar = document.createElement('div');
                            bar.className = "space-y-1";
                            bar.innerHTML = `
                                <div class="flex justify-between items-center text-[11px]">
                                    <span class="font-bold text-gray-300 uppercase">${lang}</span>
                                    <span class="text-gray-400">${count} samples (${pct}%)</span>
                                </div>
                                <div class="w-full bg-gray-900 rounded-full h-1.5 border border-gray-800">
                                    <div class="bg-brand-500 h-1 rounded-full" style="width: ${pct}%"></div>
                                </div>
                            `;
                            langContainer.appendChild(bar);
                        });
                    } else {
                        document.getElementById('stats-datasets').innerText = "0";
                        document.getElementById('stats-size').innerText = "0.00 MB";
                        document.getElementById('stats-tokens').innerText = "0";
                        document.getElementById('stats-dup').innerText = "0.0%";
                        document.getElementById('stats-qc').innerText = "0.0%";
                        document.getElementById('lang-bar-container').innerHTML = '<p class="text-xs text-gray-500 italic">No processed language statistics found.</p>';
                    }
                });
        }

        function addDatasetLink() {
            const input = document.getElementById('new-dataset-url');
            const url = input.value.trim();
            if (!url) return;
            
            fetch('/api/add_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            })
            .then(res => res.json())
            .then(data => {
                alert("Successfully added dataset URL directly to links.txt!");
                input.value = '';
                fetchDatasets();
            });
        }

        // 4. Fetch the visualization details of the model weights
        function fetchNeuronStates() {
            fetch('/api/neuron_states')
                .then(res => {
                    if (!res.ok) throw new Error("No weight data found");
                    return res.json();
                })
                .then(data => {
                    const grid = document.getElementById('neuron-grid');
                    grid.innerHTML = '';
                    
                    data.grid.forEach(val => {
                        const dot = document.createElement('div');
                        dot.className = val >= 0 
                            ? "aspect-square rounded-md bg-brand-500 shadow-sm shadow-brand-500/50" 
                            : "aspect-square rounded-md bg-indigo-950/80 border border-gray-800";
                        grid.appendChild(dot);
                    });
                })
                .catch(() => {
                    const grid = document.getElementById('neuron-grid');
                    grid.innerHTML = '';
                    for (let i = 0; i < 64; i++) {
                        const dot = document.createElement('div');
                        dot.className = "aspect-square rounded-md bg-indigo-950/80 border border-gray-800";
                        grid.appendChild(dot);
                    }
                });
        }

        // 5. Training execution
        function startTraining() {
            const steps = document.getElementById('train-steps').value;
            const checkboxes = document.querySelectorAll('#dataset-checklist input[type="checkbox"]:checked');
            const selected_repos = Array.from(checkboxes).map(cb => cb.value);
            
            const payload = {
                steps: steps,
                selected_repos: selected_repos.length > 0 ? selected_repos : null
            };
            
            fetch('/api/start_train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    fetchStatus();
                    startLogging();
                }
            });
        }

        function stopTraining() {
            fetch('/api/stop_train', { method: 'POST' })
                .then(() => {
                    fetchStatus();
                    stopLogging();
                });
        }

        function startLogging() {
            const logBox = document.getElementById('console-logs');
            const statusText = document.getElementById('train-status-text');
            
            statusText.className = "text-brand-400 flex items-center space-x-1";
            statusText.innerHTML = `
                <span class="w-2 h-2 rounded-full bg-brand-500 animate-ping"></span>
                <span>Active Pre-Training</span>
            `;
            
            if (logInterval) clearInterval(logInterval);
            
            logInterval = setInterval(() => {
                fetch('/api/logs')
                    .then(res => res.json())
                    .then(data => {
                        logBox.innerText = data.logs.join('\\n');
                        logBox.scrollTop = logBox.scrollHeight;
                        
                        document.getElementById('metric-epoch').innerText = data.epoch;
                        document.getElementById('metric-step').innerText = data.step;
                        document.getElementById('metric-loss').innerText = data.loss > 0 ? data.loss : "0.0000";
                        document.getElementById('metric-speed').innerText = data.speed + " tok/s";
                        
                        if (!data.is_training) {
                            clearInterval(logInterval);
                            logInterval = null;
                            fetchStatus();
                            fetchDatasets();
                            fetchNeuronStates();
                            
                            statusText.className = "text-gray-400 flex items-center space-x-1";
                            statusText.innerHTML = `
                                <span class="w-2 h-2 rounded-full bg-gray-500"></span>
                                <span>Completed / Idle</span>
                            `;
                        }
                    });
            }, 1000);
        }

        function stopLogging() {
            if (logInterval) {
                clearInterval(logInterval);
                logInterval = null;
            }
        }

        // 6. Causal Chat Generation
        function handleChatKey(event) {
            if (event.key === "Enter") {
                sendChatMessage();
            }
        }

        function sendChatMessage() {
            const input = document.getElementById('chat-input');
            const text = input.value.trim();
            if (!text) return;
            
            const feed = document.getElementById('chat-messages');
            
            const userMsg = document.createElement('div');
            userMsg.className = "flex items-start space-x-3 justify-end";
            userMsg.innerHTML = `
                <div class="bg-brand-600 text-white rounded-2xl px-4 py-2.5 text-sm max-w-[85%] text-left">
                    ${text}
                </div>
                <div class="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-xs font-bold">ME</div>
            `;
            feed.appendChild(userMsg);
            feed.scrollTop = feed.scrollHeight;
            
            input.value = '';
            
            const thinkingMsg = document.createElement('div');
            thinkingMsg.className = "flex items-start space-x-3";
            thinkingMsg.id = "ai-thinking-placeholder";
            thinkingMsg.innerHTML = `
                <div class="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-xs font-bold">AI</div>
                <div class="bg-gray-800 rounded-2xl px-4 py-2.5 text-sm max-w-[85%] text-gray-400 italic">
                    <i class="fa-solid fa-circle-notch animate-spin mr-1.5 text-brand-500"></i>AI is computing causal next-token weights...
                </div>
            `;
            feed.appendChild(thinkingMsg);
            feed.scrollTop = feed.scrollHeight;
            
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: text, temperature: 0.7, max_tokens: 45 })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('ai-thinking-placeholder').remove();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                const aiResponse = document.createElement('div');
                aiResponse.className = "flex items-start space-x-3";
                aiResponse.innerHTML = `
                    <div class="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-xs font-bold">AI</div>
                    <div class="bg-gray-800 rounded-2xl px-4 py-2.5 text-sm max-w-[85%] leading-relaxed select-text">
                        ${data.response}
                    </div>
                `;
                feed.appendChild(aiResponse);
                feed.scrollTop = feed.scrollHeight;
                
                const altBox = document.getElementById('alternatives-box');
                altBox.innerHTML = '';
                
                data.alternatives.forEach(alt => {
                    const pct = (alt.probability * 100).toFixed(1);
                    const div = document.createElement('div');
                    div.className = "space-y-1 text-xs";
                    div.innerHTML = `
                        <div class="flex justify-between items-center text-[11px]">
                            <span class="font-mono text-gray-300 font-bold">${alt.rank}. '${alt.token.replace('\\n', ' ')}'</span>
                            <span class="text-brand-400 font-bold">${pct}%</span>
                        </div>
                        <div class="w-full bg-gray-900 rounded-full h-1">
                            <div class="bg-brand-500 h-1 rounded-full" style="width: ${pct}%"></div>
                        </div>
                    `;
                    altBox.appendChild(div);
                });
                
                fetchNeuronStates();
            })
            .catch(() => {
                document.getElementById('ai-thinking-placeholder').remove();
                alert("Web request failed. Ensure your Flask server is running.");
            });
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    """Serves the main embedded HTML dashboard."""
    return render_template_string(HTML_TEMPLATE)

def run_app():
    """Launches the server."""
    print("="*60)
    print("      N1Bit-ARM64 AI WEB DASHBOARD SUCCESSFULLY STARTED!")
    print("="*60)
    print(" To access your beautiful 1-bit graphical UI on your phone:")
    print("   1. Click on the web link below.")
    print("   2. Or if running on phone, open: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    run_app()
