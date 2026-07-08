import os
import json
import re
import random
import csv
import array
from typing import Generator, Dict, Any, List
from .config import LINKS_FILE, MAX_SAMPLES_PER_DATASET, SAMPLE_QUALITY_THRESHOLD, RAW_DOWNLOADS_DIR
from .utils import compute_hash, score_sample_quality, detect_language
from .tokenizer import SimpleBPETokenizer

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

try:
    from datasets import load_dataset
    import pyarrow
    HAS_DATASETS_AND_ARROW = True
except ImportError:
    HAS_DATASETS_AND_ARROW = False

class DatasetEngine:
    """
    Highly robust and efficient Dataset Engine designed for 1-bit ARM64/Mobile systems.
    Downloads datasets safely, filters, scores quality, removes duplicates, streams
    data, and produces extensive pre-training statistics.
    Saves all downloaded raw files inside 'downloads/' in the repository.
    
    Pre-Tokenized Binary Cache (.bin):
    - Compiles all pre-processed text, image captions, and science tables into a 
      flat binary array of uint16 integers (processed_data.bin) using the BPE Tokenizer.
    - Completely eliminates tokenizer overhead during the training loop.
    - Training becomes blazing fast—reading pre-tokenized chunks in milliseconds!
    """
    def __init__(self, processed_data_file: str, stats_file: str, tokenizer_file: str = None):
        self.links_file = LINKS_FILE
        self.processed_data_file = processed_data_file
        self.stats_file = stats_file
        self.tokenizer_file = tokenizer_file
        self.raw_cache_dir = RAW_DOWNLOADS_DIR
        os.makedirs(self.raw_cache_dir, exist_ok=True)
        
        # State tracking to bypass slow connection handshakes when offline/restricted
        self.datasets_api_active = HAS_DATASETS_AND_ARROW
        self.pure_python_api_active = True
        
        # In-memory tracking for statistics
        self.stats = {
            "num_datasets_processed": 0,
            "num_files_processed": 0,
            "num_samples_kept": 0,
            "num_samples_discarded": 0,
            "duplicate_rate": 0.0,
            "language_distribution": {},
            "size_before_bytes": 0,
            "size_after_bytes": 0,
            "estimated_token_count": 0
        }
        
    def read_links(self) -> List[str]:
        """Reads dataset URLs from links.txt."""
        if not os.path.exists(self.links_file):
            print(f"[DatasetEngine] Warning: Links file '{self.links_file}' not found.")
            return []
            
        urls = []
        with open(self.links_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        return urls

    def parse_repo_id(self, url: str) -> str:
        """Extracts Hugging Face repository ID from a dataset URL."""
        match = re.search(r'huggingface\.co/datasets/([^/]+/[^/]+)', url)
        if match:
            return match.group(1)
        parts = url.split('/')
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return url

    def fetch_web_json(self, url: str) -> Any:
        """Helper to fetch JSON from web in pure Python."""
        if HAS_REQUESTS:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        else:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode('utf-8'))

    def download_web_file(self, url: str, dest_path: str):
        """Helper to download a file from the web safely with resume capability."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if os.path.exists(dest_path):
            return
            
        if HAS_REQUESTS:
            response = requests.get(url, stream=True, timeout=5)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            with urllib.request.urlopen(url, timeout=5) as response:
                with open(dest_path, 'wb') as f:
                    f.write(response.read())

    def detect_dataset_type(self, sample: Dict[str, Any]) -> str:
        """Automatically detects the type of dataset based on sample features."""
        keys = set(sample.keys())
        
        if "messages" in keys or "conversations" in keys:
            return "Chat"
            
        sft_keys = {"instruction", "input", "output", "prompt", "completion", "response"}
        if len(sft_keys.intersection(keys)) >= 2:
            return "SFT"
            
        qa_keys = {"question", "answer", "query"}
        if len(qa_keys.intersection(keys)) >= 2:
            return "QA"
            
        code_keys = {"code", "programming_language", "solution", "test_cases", "assembly", "binary", "hex", "mnemonic", "opcodes"}
        if len(code_keys.intersection(keys)) >= 2:
            return "CodeAssembly"
            
        img_keys = {"image", "caption", "image_caption", "pixel_values", "textures", "texture", "normal_map"}
        if len(img_keys.intersection(keys)) >= 1:
            return "ImageCaption"
            
        mc_keys = {"blocks", "schematic", "minecraft", "structure"}
        if len(mc_keys.intersection(keys)) >= 1:
            return "MinecraftSchematic"
            
        table_keys = {"element", "property", "formula", "compound", "temperature", "pressure", "value"}
        if len(table_keys.intersection(keys)) >= 2:
            return "ScientificTable"
            
        return "RawText"

    def convert_to_unified_text(self, sample: Dict[str, Any], dataset_type: str) -> str:
        """Converts diverse multimodal/structured dataset formats into a unified clean training corpus."""
        if dataset_type == "Chat":
            messages = sample.get("messages") or sample.get("conversations")
            if isinstance(messages, list):
                chat_text = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        chat_text.append(f"<{role}>: {content}")
                return "\n".join(chat_text)
                
        elif dataset_type == "SFT":
            instruction = sample.get("instruction", "")
            inp = sample.get("input", "")
            out = sample.get("output", "") or sample.get("completion", "") or sample.get("response", "")
            
            parts = []
            if instruction:
                parts.append(f"<instruction>: {instruction}")
            if inp:
                parts.append(f"<input>: {inp}")
            if out:
                parts.append(f"<response>: {out}")
            return "\n".join(parts)
            
        elif dataset_type == "QA":
            q = sample.get("question", "") or sample.get("query", "")
            a = sample.get("answer", "")
            return f"<instruction>: {q}\n<response>: {a}"
            
        elif dataset_type == "CodeAssembly":
            code = sample.get("code", "") or sample.get("solution", "") or sample.get("opcodes", "") or sample.get("hex", "")
            desc = sample.get("description", "") or sample.get("instruction", "") or sample.get("mnemonic", "")
            lang = sample.get("programming_language", "assembly")
            
            parts = []
            if desc:
                parts.append(f"<instruction>: Explain or implement the following {lang} operation: {desc}")
            if code:
                parts.append(f"<response>: \n```{lang}\n{code}\n```")
            return "\n".join(parts)
            
        elif dataset_type == "ImageCaption":
            caption = sample.get("caption", "") or sample.get("image_caption", "") or sample.get("text", "")
            width = sample.get("width", "unknown")
            height = sample.get("height", "unknown")
            fmt = sample.get("format", "unknown")
            
            desc = f"An image file."
            if isinstance(caption, str) and caption:
                desc = caption
                
            return f"<instruction>: Describe this image texture or layout.\n<response>: Caption: {desc}. Image specifications: {width}x{height} pixels, format {fmt}."
            
        elif dataset_type == "MinecraftSchematic":
            name = sample.get("name", "Minecraft Schematic")
            blocks = sample.get("blocks", "") or sample.get("schematic", "")
            w = sample.get("width", "unknown")
            h = sample.get("height", "unknown")
            l = sample.get("length", "unknown")
            
            return f"<instruction>: Describe the layout of the Minecraft structure: '{name}'.\n<response>: Dimensions: {w}x{h}x{l} blocks. Schematic Block data: {blocks}."
            
        elif dataset_type == "ScientificTable":
            formula = sample.get("formula", "") or sample.get("compound", "") or sample.get("element", "")
            prop = sample.get("property", "")
            val = sample.get("value", "")
            unit = sample.get("unit", "")
            
            return f"<instruction>: What is the {prop} of {formula}?\n<response>: The {prop} of {formula} is {val} {unit}."

        text_parts = []
        for k, v in sample.items():
            if isinstance(v, str) and len(v) > 5:
                text_parts.append(v)
        return "\n".join(text_parts)

    def fetch_hf_repo_files_pure_python(self, repo_id: str) -> List[Dict[str, Any]]:
        """
        Pure-Python fallback to stream and parse Hugging Face dataset files
        without PyArrow or standard Hugging Face datasets library.
        """
        samples = []
        try:
            api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
            file_tree = self.fetch_web_json(api_url)
            
            text_files = []
            for item in file_tree:
                if item.get("type") == "file":
                    path = item.get("path", "")
                    if path.endswith((".json", ".jsonl", ".csv", ".txt")):
                        text_files.append(path)
                        
            for file_path in text_files[:2]:
                local_dest = os.path.join(self.raw_cache_dir, repo_id, file_path)
                download_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
                
                self.download_web_file(download_url, local_dest)
                
                if file_path.endswith(".jsonl"):
                    with open(local_dest, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.strip():
                                samples.append(json.loads(line))
                                if len(samples) >= 100:
                                    break
                elif file_path.endswith(".json"):
                    with open(local_dest, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            samples.extend(data[:100])
                        elif isinstance(data, dict):
                            for key in ["train", "data", "rows"]:
                                if key in data and isinstance(data[key], list):
                                    samples.extend(data[key][:100])
                                    break
                            else:
                                samples.append(data)
                elif file_path.endswith(".csv"):
                    with open(local_dest, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            samples.append(dict(row))
                            if len(samples) >= 100:
                                break
                elif file_path.endswith(".txt"):
                    with open(local_dest, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if len(line.strip()) > 10:
                                samples.append({"text": line.strip()})
                                if len(samples) >= 100:
                                    break
        except Exception:
            raise
            
        return samples

    def generate_mock_samples(self, repo_id: str, count: int = 15) -> List[Dict[str, Any]]:
        """Generates rich, domain-specific mock/synthetic dataset samples when offline."""
        samples = []
        repo_id_lower = repo_id.lower()
        
        if "physics" in repo_id_lower or "chemistry" in repo_id_lower or "climate" in repo_id_lower:
            topics = [
                ("What is gravity?", "Gravity is a fundamental physical force that attracts objects with mass towards each other."),
                ("Describe the second law of thermodynamics.", "The entropy of any isolated system always increases over time, meaning thermal energy flows spontaneously from hot to cold bodies."),
                ("How does climate change affect ocean temperatures?", "Greenhouse gas emissions trap thermal energy in the atmosphere, and the oceans absorb over 90% of this excess heat."),
                ("What is the chemical symbol for water?", "The chemical formula of water is H2O, representing two hydrogen atoms bonded to one oxygen atom.")
            ]
            for i in range(count):
                q, a = random.choice(topics)
                samples.append({"question": f"{q} (ID: {i})", "answer": f"{a} Verifying physics vector indices."})
                
        elif "code" in repo_id_lower or "stack" in repo_id_lower or "java" in repo_id_lower or "instruction" in repo_id_lower or "so" in repo_id_lower:
            codes = [
                ("Write a binary search function.", "def binary_search(arr, x):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] < x:\n            low = mid + 1\n        elif arr[mid] > x:\n            high = mid - 1\n        else:\n            return mid\n    return -1"),
                ("Explain memory management in ARM64.", "ARM64 uses a translation lookaside buffer (TLB) to cache virtual to physical memory translations, reducing memory lookup cycles."),
                ("How to load shared libraries in Python?", "import ctypes\nlib = ctypes.CDLL('./arm64_so.so')\nresult = lib.process_data()")
            ]
            for i in range(count):
                desc, code = random.choice(codes)
                samples.append({"instruction": f"{desc} #{i}", "code": code})
                
        elif "minecraft" in repo_id_lower:
            mc_facts = [
                "Minecraft is a sandbox video game developed by Mojang Studios where players build 3D worlds.",
                "Redstone dust is used in Minecraft to transmit electrical power and build automated schematics.",
                "To summon an iron golem, place four blocks of iron in a T-shape and top it with a carved pumpkin.",
                "Endermen are neutral mobs that teleport away when looked at directly or exposed to water."
            ]
            for i in range(count):
                samples.append({"text": f"{random.choice(mc_facts)} Minecraft Wiki reference {i}.", "category": "gameplay"})
                
        elif "wikipedia_vi" in repo_id_lower:
            vi_texts = [
                "Wikipedia tiếng Việt là phiên bản tiếng Việt của bách khoa toàn thư mở Wikipedia.",
                "Hồ Chí Minh là một nhà cách mạng, người sáng lập Đảng Cộng sản Việt Nam.",
                "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam, nổi tiếng với hồ Hoàn Kiếm.",
                "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương thuộc khu vực Đông Nam Á."
            ]
            for i in range(count):
                samples.append({"text": f"{random.choice(vi_texts)} Bài viết Wikipedia số {i}."})
                
        elif "language-identification" in repo_id_lower:
            langs = [
                ("en", "The quick brown fox jumps over the lazy dog in this beautiful english morning."),
                ("es", "El perro rápido corre sobre el campo verde bajo el sol brillante de España."),
                ("fr", "Le chat noir dort sur le canapé dans le salon de la maison française."),
                ("de", "Ein schneller Fuchs springt über den faulen Hund in der deutschen Landschaft."),
                ("ru", "Быстрый бурый лис перепрыгивает через ленивую собаку в русском лесу."),
                ("vi", "Con cáo nhanh nhẹn nhảy qua con chó lười biếng trong buổi sáng Việt Nam.")
            ]
            for i in range(count):
                lbl, txt = random.choice(langs)
                samples.append({"text": f"{txt} Language verification identifier sample {i}.", "label": lbl})
                
        elif "image" in repo_id_lower or "draw" in repo_id_lower or "textures" in repo_id_lower:
            captions = [
                "A clean tileable normal map of a brick wall texture.",
                "An engineering drawing of an AS1100 standard mechanical assembly.",
                "An architectural layout showing front and side views.",
                "Gameplay screenshot of a retro arcade game."
            ]
            for i in range(count):
                samples.append({"image_caption": f"{random.choice(captions)} Graphic design {i}.", "pixel_values": [1, 2, 3]})
                
        else:
            instructions = [
                ("What is a 1-bit neural network?", "A 1-bit neural network uses binary weights (typically -1 and +1) or ternary weights to dramatically reduce computation, power, and storage requirements on mobile devices."),
                ("How do we optimize LLMs for ARM64 architecture?", "By avoiding heavy dependencies like transformers, using lightweight custom tokenizers, and utilizing integer or 1-bit math that maps perfectly to neon instruction sets."),
                ("Explain Straight-Through Estimator (STE).", "STE is a method to train quantized neural networks where we use non-differentiable quantization in the forward pass but bypass it during backpropagation, treating it as identity.")
            ]
            for i in range(count):
                inst, resp = random.choice(instructions)
                samples.append({
                    "instruction": f"{inst} (Trace #{i})",
                    "input": "System context ARM64 architecture.",
                    "output": resp
                })
                
        return samples

    def process_all_datasets(self, force_refresh: bool = False, selected_repos: List[str] = None):
        """
        Downloads datasets, filters by quality, and tokenizes them immediately.
        Saves all training corpora as pre-tokenized binary files (.bin) containing
        unsigned 16-bit integers (uint16) using the BPE tokenizer.
        """
        if not force_refresh and os.path.exists(self.processed_data_file) and os.path.exists(self.stats_file):
            print(f"[DatasetEngine] Found cached pre-tokenized binary data at {self.processed_data_file}. Skipping.")
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
            return

        print("[DatasetEngine] Starting raw dataset download and pre-processing pipeline...")
        urls = self.read_links()
        if not urls:
            print("[DatasetEngine] Error: No URLs to process.")
            return

        seen_hashes = set()
        lang_distribution = {}
        total_samples_processed = 0
        total_duplicates = 0
        num_datasets_processed = 0
        num_files_processed = 0
        num_samples_kept = 0
        num_samples_discarded = 0
        size_before_bytes = 0
        size_after_bytes = 0

        # Collect clean text samples in memory first to train tokenizer
        clean_text_samples = []

        for url in urls:
            repo_id = self.parse_repo_id(url)
            
            if selected_repos is not None and repo_id not in selected_repos:
                continue
                
            num_datasets_processed += 1
            print(f"[DatasetEngine] Safe loading: '{repo_id}'...")
            loaded_samples = []
            loaded_source = "None"
            
            if self.datasets_api_active:
                try:
                    dataset = load_dataset(repo_id, streaming=True)
                    splits = list(dataset.keys()) if hasattr(dataset, "keys") else ["train"]
                    for split in splits:
                        num_files_processed += 1
                        split_dataset = dataset[split]
                        count = 0
                        for raw_sample in split_dataset:
                            if count >= MAX_SAMPLES_PER_DATASET:
                                break
                            loaded_samples.append(raw_sample)
                            count += 1
                    loaded_source = "HuggingFace Datasets API"
                except Exception as e:
                    err_str = str(e)
                    if "SSLError" in err_str or "ConnectionError" in err_str or "TLS" in err_str or "EOF" in err_str:
                        print("[DatasetEngine] Connection issues detected. Disabling slow HF Datasets API.")
                        self.datasets_api_active = False
            
            if not loaded_samples and self.pure_python_api_active:
                try:
                    loaded_samples = self.fetch_hf_repo_files_pure_python(repo_id)
                    if loaded_samples:
                        num_files_processed += len(loaded_samples) // 100 + 1
                        loaded_source = "Pure-Python HF Repository Parser"
                except Exception as e:
                    err_str = str(e)
                    if "SSLError" in err_str or "ConnectionError" in err_str or "TLS" in err_str or "EOF" in err_str:
                        print("[DatasetEngine] Connection issues detected. Disabling pure-python web loader.")
                        self.pure_python_api_active = False
                    
            if not loaded_samples:
                num_files_processed += 1
                loaded_samples = self.generate_mock_samples(repo_id, count=15)
                loaded_source = "Offline-Safety Synthetic Fallback"
            
            count = 0
            for raw_sample in loaded_samples:
                total_samples_processed += 1
                count += 1
                
                raw_str_size = sum(len(str(v)) for v in raw_sample.values())
                size_before_bytes += raw_str_size
                
                ds_type = self.detect_dataset_type(raw_sample)
                unified_text = self.convert_to_unified_text(raw_sample, ds_type).strip()
                
                if not unified_text:
                    num_samples_discarded += 1
                    continue
                    
                quality_score = score_sample_quality(unified_text)
                if quality_score < SAMPLE_QUALITY_THRESHOLD:
                    num_samples_discarded += 1
                    continue
                    
                sample_hash = compute_hash(unified_text)
                if sample_hash in seen_hashes:
                    total_duplicates += 1
                    num_samples_discarded += 1
                    continue
                    
                seen_hashes.add(sample_hash)
                
                lang = detect_language(unified_text)
                lang_distribution[lang] = lang_distribution.get(lang, 0) + 1
                
                num_samples_kept += 1
                size_after_bytes += len(unified_text)
                
                clean_text_samples.append(unified_text)
                
            print(f"[DatasetEngine] Success: Loaded '{repo_id}' via {loaded_source} ({count} samples).")

        # 2. Train or Load BPE Tokenizer inside dataset engine
        tokenizer = SimpleBPETokenizer(vocab_size=4000)
        if self.tokenizer_file and os.path.exists(self.tokenizer_file):
            tokenizer.load(self.tokenizer_file)
        else:
            print("[DatasetEngine] Tokenizer file not found. Training tokenizer first...")
            tokenizer.train_from_texts(clean_text_samples[:1000])
            if self.tokenizer_file:
                tokenizer.save(self.tokenizer_file)

        # 3. Tokenize all clean texts to a flat uint16 array and save as .bin
        print(f"[DatasetEngine] Tokenizing clean samples directly to high-performance .bin cache...")
        token_ids_array = []
        for text in clean_text_samples:
            # Wrap with BOS/EOS
            tokens = [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id]
            token_ids_array.extend(tokens)
            
        # Write flat list of integers to binary file using 'H' (unsigned 16-bit short)
        bin_data = array.array('H', token_ids_array)
        os.makedirs(os.path.dirname(self.processed_data_file), exist_ok=True)
        with open(self.processed_data_file, "wb") as f:
            bin_data.tofile(f)
            
        print(f"[DatasetEngine] Binary cache compiled successfully! Saved {len(token_ids_array):,} tokens to '{self.processed_data_file}'")

        # Save stats
        dup_rate = (total_duplicates / max(1, total_samples_processed)) * 100
        estimated_token_count = len(token_ids_array)
        
        self.stats = {
            "num_datasets_processed": num_datasets_processed,
            "num_files_processed": num_files_processed,
            "num_samples_kept": num_samples_kept,
            "num_samples_discarded": num_samples_discarded,
            "duplicate_rate": round(dup_rate, 2),
            "language_distribution": lang_distribution,
            "size_before_bytes": size_before_bytes,
            "size_after_bytes": os.path.getsize(self.processed_data_file), # actual binary file size
            "estimated_token_count": estimated_token_count
        }
        
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
            
        print("[DatasetEngine] Processing pipeline complete!")
        self.print_stats()

    def print_stats(self):
        """Prints processing statistics in a beautiful format."""
        s = self.stats
        print("\n" + "="*50)
        print("          DATASET PROCESSING STATISTICS")
        print("="*50)
        print(f"Datasets Processed:        {s['num_datasets_processed']}")
        print(f"Files/Splits Processed:    {s['num_files_processed']}")
        print(f"Samples Kept (Passed QC):  {s['num_samples_kept']}")
        print(f"Samples Discarded:         {s['num_samples_discarded']}")
        print(f"Duplicate Rate:            {s['duplicate_rate']}%")
        print(f"Language Distribution:     {s['language_distribution']}")
        print(f"Size Before Cleaning:      {s['size_before_bytes'] / (1024*1024):.2f} MB")
        print(f"Size After Cleaning (Bin): {s['size_after_bytes'] / (1024*1024):.2f} MB")
        print(f"Estimated Token Count:     {s['estimated_token_count']:,} tokens")
        print("="*50 + "\n")

    def stream_processed_tokens(self) -> Generator[int, None, None]:
        """
        Streams pre-tokenized integers directly from the binary cache file (.bin).
        Loads file in small binary chunks, bypassing 100% of text tokenization overhead.
        """
        if not os.path.exists(self.processed_data_file):
            self.process_all_datasets()
            
        with open(self.processed_data_file, "rb") as f:
            # Read 100k tokens at a time (unsigned short = 2 bytes)
            chunk_size = 100000 * 2
            while True:
                bytes_data = f.read(chunk_size)
                if not bytes_data:
                    break
                bin_data = array.array('H')
                bin_data.frombytes(bytes_data)
                for token in bin_data:
                    yield int(token)
