import re
import hashlib
import platform
import subprocess
import sys

# Unicode ranges for non-Latin script detection
CYRILLIC_RE = re.compile(r'[\u0400-\u04FF]')
CJK_RE = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]')
VIETNAMESE_RE = re.compile(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]')

# Stop words lists for simple language detection
STOP_WORDS = {
    "en": {"the", "and", "of", "to", "a", "is", "in", "that", "it", "you", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"},
    "es": {"el", "la", "de", "que", "un", "una", "y", "en", "es", "no", "se", "por", "para", "con", "los", "las", "su", "al", "como", "del"},
    "fr": {"le", "la", "de", "et", "un", "une", "en", "est", "que", "qui", "dans", "pour", "par", "sur", "plus", "avec", "ce", "les", "des", "se"},
    "de": {"der", "die", "das", "und", "ist", "in", "zu", "den", "von", "mit", "dem", "des", "ein", "eine", "nicht", "sich", "auf", "für", "im", "auch"},
    "vi": {"và", "là", "của", "trong", "có", "được", "cho", "với", "không", "một", "các", "nhưng", "đã", "khi", "những", "này", "theo", "để", "tại", "về"},
    "ru": {"и", "в", "не", "на", "я", "быть", "он", "с", "что", "а", "по", "но", "они", "мы", "это", "тот", "как", "к", "у", "же"}
}

def detect_language(text: str) -> str:
    """
    Detects the language of a given text using character sets and high-frequency stop words.
    Extremely fast, pure Python, lightweight, and requires no external packages.
    """
    text_lower = text.lower()
    
    # Check for Chinese/Japanese/Korean (CJK)
    if len(CJK_RE.findall(text)) > len(text) * 0.1:
        return "zh_ja"
    
    # Check for Russian/Cyrillic
    if len(CYRILLIC_RE.findall(text)) > len(text) * 0.1:
        return "ru"
        
    # Check for Vietnamese accents
    if len(VIETNAMESE_RE.findall(text_lower)) > len(text) * 0.02:
        return "vi"
        
    # Count stop words
    words = set(re.findall(r'[a-z]+', text_lower))
    best_lang = "en"
    max_matches = 0
    
    for lang, sw in STOP_WORDS.items():
        matches = len(words.intersection(sw))
        if matches > max_matches:
            max_matches = matches
            best_lang = lang
            
    # Default to English if no clear matches
    return best_lang

def compute_hash(text: str) -> str:
    """
    Computes MD5 hash of a normalized string for fast duplicate detection.
    """
    # Normalize text: lowercase, remove non-alphanumeric and extra spaces
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    # Return MD5 hex digest
    return hashlib.md5(normalized.encode('utf-8', errors='ignore')).hexdigest()

def score_sample_quality(text: str) -> float:
    """
    Scores the quality of a text sample.
    Returns a score between 0.0 and 1.0.
    Considers:
      - Length and word count (penalizes extremely short/long or empty texts).
      - Vocabulary variety (penalizes highly repetitive spam/gibberish).
      - Alphanumeric to punctuation/special char ratio (penalizes binary dumps/corrupted data).
    """
    if not text or len(text.strip()) == 0:
        return 0.0
        
    text_len = len(text)
    
    # 1. Length penalty
    # Optimal length is between 50 and 50,000 characters
    if text_len < 10:
        return 0.0
    elif text_len < 50:
        len_score = 0.4
    elif text_len > 100000:
        len_score = 0.5  # Penalize extremely long files/logs
    else:
        len_score = 1.0
        
    # 2. Vocabulary variety (Repetitiveness)
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    
    unique_words_ratio = len(set(words)) / len(words)
    # A text with very low unique words ratio is highly repetitive (e.g. repeated spam or template errors)
    # But a very short text naturally has high unique words, so we combine it
    if len(words) > 20:
        if unique_words_ratio < 0.2:
            rep_score = 0.1  # Highly repetitive
        elif unique_words_ratio < 0.4:
            rep_score = 0.6
        else:
            rep_score = 1.0
    else:
        rep_score = 1.0
        
    # 3. Alphanumeric density (checks for binary corruption, memory dumps, or excessively scrambled lines)
    alnum_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    alnum_ratio = alnum_chars / text_len
    
    # Text should be mostly alphanumeric and spaces
    if alnum_ratio < 0.5:
        alnum_score = 0.1  # Likely raw binaries, hex dumps, or heavy markup corruption
    elif alnum_ratio < 0.7:
        alnum_score = 0.6
    else:
        alnum_score = 1.0
        
    # Calculate overall weighted score
    overall_score = (len_score * 0.3) + (rep_score * 0.4) + (alnum_score * 0.3)
    return overall_score

def optimize_environment():
    """
    Check if we are in PC environment and suggest/install x86 optimizations
    such as CUDA support, optimized kernels, or pyav/image packages.
    """
    from .config import is_pc_environment
    
    if is_pc_environment():
        print("[System Info] Detected PC/x86 environment. Accelerating with CUDA and high-performance libraries if available.")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[System Info] CUDA detected: {torch.cuda.get_device_name(0)}. Using GPU acceleration.")
            else:
                print("[System Info] CUDA is not available. Using multi-threaded CPU processing.")
        except ImportError:
            print("[System Info] PyTorch not installed. Installing standard PyTorch for x86 architecture.")
            try:
                # Install torch for CPU/CUDA based on system
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--break-system-packages"])
                print("[System Info] PyTorch installed successfully.")
            except Exception as e:
                print(f"[System Info] Failed to install PyTorch automatically: {e}")
    else:
        print("[System Info] Detected ARM64 / Mobile Environment. Running in low-power, lightweight optimization mode.")
