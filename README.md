# N1Bit-ARM64

An ultra-low-power, maximum-speed **1-Bit AI Architecture** (BitNet b1.58 style) written entirely from scratch in Python and NumPy. This engine is specifically optimized for **ARM64 / Mobile environments** (like Termux on Android), with full dynamic fallbacks to accelerate with PyTorch and CUDA when run in PC environments.

---

## 🚀 Key Features

### 1. Zero Heavy Dependencies (No Transformers or Tokenizers C-Extensions)
Standard LLM tokenizers (like Tiktoken or SentencePiece) and frameworks (like Hugging Face Transformers) require x86 or heavy pre-compiled C++ libraries. These are notoriously slow or impossible to load and build on mobile devices. 
- **Custom BPE Tokenizer:** Written completely in **pure Python**, featuring extremely fast training and encoding, zero external C/C++ compiled dependencies, and full save/load support.
- **Pure NumPy Model Engine:** Allows running fully quantized, 1-bit autoregressive inference and Recurrent network training with **zero dependencies on PyTorch**—minimizing RAM footprint and power consumption on mobile phones.

### 2. 1-Bit AI Architecture (BitNet b1.58 implementation)
- **Ternary Weight Quantization:** Weights are quantized to ternary values $\{-1, 0, 1\}$ using a scaling factor $\beta = \text{mean}(|W|)$ to mimic the state-of-the-art BitNet b1.58.
- **8-Bit Activation Quantization:** Activations are dynamically scaled and quantized to 8-bit integers to reduce thermal output, save memory, and accelerate matrix dot products.
- **Straight-Through Estimator (STE):** Training employs STE to route continuous gradients around the non-differentiable quantization function.

### 3. Dynamic PC vs. ARM64 Environment Detection
- Automatically checks if you are running on a PC (x86_64) or if CUDA is available.
- Suggests/installs standard accelerated libraries like **PyTorch with CUDA** to train the full causal **BitTransformerLM** model at lightning speed.
- On ARM64 (mobile) or environments without PyTorch, it automatically falls back to training our custom, highly optimized **NumPyBitRNNLM** recurrent sequence model using optimized NumPy matrix operations.

### 4. Advanced Streaming Dataset Engine
Designed to process the 52 datasets specified in `links.txt` without bloating RAM:
- **Streaming Mode:** Uses Hugging Face datasets streaming (`streaming=True`) to read samples line-by-line with zero memory overhead.
- **Safe Loader & Resume:** Gracefully skips deleted, private, or gated datasets, handles network glitches, and resumes interrupted downloads natively.
- **Offline Synthetic Fallback:** If internet access is blocked (e.g. in restricted sandboxes or offline devices), the engine automatically spins up a highly realistic domain-specific synthetic generator for each of the 52 datasets to keep training and testing 100% operational.
- **Quality Scoring (QC):** Filters corrupt samples, raw binary memory dumps, or repetitive garbage based on length, alphanumeric density, and vocabulary variety.
- **Fingerprint De-duplication:** Uses an extremely fast, low-memory MD5/SHA-256 string normalizer to detect and eliminate duplicate samples across all processed datasets.
- **Unified Training Corpus:** Formats SFT instruction-tuning, QA, multi-turn chat templates, raw code, and multimodal text captions into a clean, unified `<instruction>/<response>` structure.
- **Reusable Pre-training Cache:** Stores processed text in a high-speed `.jsonl` cache, skipping the extraction and QC loops entirely on subsequent runs.
- **Detailed Pre-training Statistics:** Generates and exports extensive statistics (saved to `cache/dataset_stats.json`) including:
  - Number of datasets processed
  - Number of files processed
  - Number of samples kept vs. discarded
  - Duplicate rate (%)
  - Language distribution (English, Spanish, French, German, Russian, Vietnamese)
  - Raw vs. clean dataset sizes (MB)
  - Estimated token count

---

## 📁 Repository Structure

```bash
N1Bit-ARM64/
├── links.txt               # The ONLY source of dataset URLs (52 datasets)
├── train.py                # Pre-training pipeline CLI runner
├── interactive.py          # Interactive Chat UI (supports PyTorch & Pure NumPy engines)
├── n1bit/
│   ├── __init__.py
│   ├── config.py           # Model and dataset hyperparameters (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
│   ├── utils.py            # Simple language detection, QC scoring, MD5 hashes, and PC optimization
│   ├── tokenizer.py        # Pure-Python Byte-Pair-Encoder (BPE)
│   ├── dataset.py          # Safely streams, filters, de-duplicates, and caches 52 datasets
│   ├── model.py            # PyTorch BitTransformerLM & NumPy BitTransformerLM + NumPyBitRNNLM
│   └── trainer.py          # Manages training loop, PyTorch checkpointing, and NumPy exporting
└── cache/                  # Auto-generated (Git-ignored) pre-processing cache & weights
    ├── processed_data.jsonl# Saved clean training corpus
    ├── dataset_stats.json  # Pre-processing statistics
    ├── tokenizer.json      # Trained BPE vocabulary
    ├── model_checkpoint.pt # Saved PyTorch continuous weights
    └── numpy_weights.npz   # Exported 1-bit quantized NumPy weights
```

---

## 🚀 How to Run

### 1. Dataset Pre-processing & Pre-Training
To start the pipeline, train the tokenizer, process the 52 datasets from `links.txt`, pre-train the model, and export the quantized weights to NumPy, run:
```bash
python3 train.py
```
*Note: To run a quick test with a limited number of steps (e.g. 5 steps), pass the step count as an argument:*
```bash
python3 train.py 5
```

### 2. Interactive Chat
To run the interactive CLI and chat with your trained 1-bit AI model, run:
```bash
python3 interactive.py
```
You will be greeted with an interactive console menu:
```text
============================================================
    1-Bit AI Engine (BitNet b1.58) optimized for ARM64/Mobile
============================================================

Choose an option:
1. Run Dataset Engine & Train 1-Bit AI model
2. Run Interactive Chat (PyTorch Inference)
3. Run Interactive Chat (Pure NumPy - Ultra-Low Power Inference)
4. Display Pre-processing Statistics
5. Exit
```

---

## 📊 Pre-processing Statistics Example
When processing completes, the engine displays beautiful, comprehensive reports:
```text
==================================================
          DATASET PROCESSING STATISTICS
==================================================
Datasets Processed:        52
Files/Splits Processed:    52
Samples Kept (Passed QC):  258
Samples Discarded:         522
Duplicate Rate:            66.92%
Language Distribution:     {'en': 230, 'es': 4, 'vi': 17, 'de': 3, 'fr': 2, 'ru': 2}
Size Before Cleaning:      0.15 MB
Size After Cleaning:       0.04 MB
Estimated Token Count:     11571
==================================================
```

---

## 🛡️ License
This project is licensed under the terms of the MIT License included in this repository.
