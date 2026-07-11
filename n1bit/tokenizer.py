import os
import json
from collections import Counter
from typing import List, Dict

class SimpleBPETokenizer:
    """
    A pure-Python implementation of a Byte-Pair Encoding (BPE) Tokenizer.
    Extremely lightweight, does not require any external C/C++ compiled binaries,
    which makes it perfect for low-power ARM64/Mobile devices.
    """
    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<instruction>", "<response>"]
        
        # Initialize vocab maps
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[tuple, str] = {}
        
        # Reset vocab
        self._reset_vocab()
        
    def _reset_vocab(self):
        self.vocab = {}
        # 1. Add special tokens
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            
        # 2. Add printable characters and standard bytes (0-255)
        curr_idx = len(self.special_tokens)
        for i in range(256):
            char = chr(i)
            # Avoid overwriting special tokens or control characters if possible
            if char not in self.vocab:
                self.vocab[char] = curr_idx
                curr_idx += 1
                
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<s>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["</s>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    def train_from_texts(self, texts: List[str]):
        """
        Trains BPE on a collection of texts.
        To avoid slow execution on mobile/ARM64, we limit training sample size.
        """
        self._reset_vocab()
        
        # Sample or join text
        sample_text = " ".join(texts)[:100000]  # Limit to first 100K chars for fast training
        if not sample_text:
            return
            
        # Split text into list of lists of characters
        # Each word is split into characters with a special end-of-word marker
        words = sample_text.split()
        word_freqs = Counter(words)
        
        # Represent words as lists of chars
        splits = {word: list(word) for word in word_freqs.keys()}
        
        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            return
            
        # Perform merges iteratively
        for _ in range(num_merges):
            # Count pair frequencies
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                split = splits[word]
                for i in range(len(split) - 1):
                    pair_freqs[(split[i], split[i+1])] += freq
                    
            if not pair_freqs:
                break
                
            # Find the most common pair
            best_pair = pair_freqs.most_common(1)[0][0]
            
            # Create the new merged token
            new_token = "".join(best_pair)
            new_idx = len(self.vocab)
            
            self.vocab[new_token] = new_idx
            self.inverse_vocab[new_idx] = new_token
            self.merges[best_pair] = new_token
            
            # Apply merge to all splits
            for word in word_freqs.keys():
                split = splits[word]
                i = 0
                new_split = []
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i+1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split

    def encode(self, text: str) -> List[int]:
        """
        Encodes text into a list of token IDs.
        Protects special tokens from being broken into individual characters.
        """
        if not text:
            return []
            
        import re
        # Match special tokens
        pattern = re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
        parts = pattern.split(text)
        
        token_ids = []
        for part in parts:
            if not part:
                continue
                
            # If it is a special token, yield its ID directly!
            if part in self.vocab:
                token_ids.append(self.vocab[part])
            else:
                # Otherwise, tokenize using normal word-based BPE merges
                words = part.split(' ')
                for idx, word in enumerate(words):
                    if idx > 0:
                        token_ids.append(self.vocab.get(" ", self.unk_id))
                        
                    chars = list(word)
                    while True:
                        pairs = [(chars[i], chars[i+1]) for i in range(len(chars) - 1)]
                        if not pairs:
                            break
                            
                        best_pair = None
                        best_rank = float('inf')
                        for pair in pairs:
                            if pair in self.merges:
                                rank = list(self.merges.keys()).index(pair)
                                if rank < best_rank:
                                    best_rank = rank
                                    best_pair = pair
                                    
                        if best_pair is None:
                            break
                            
                        new_token = self.merges[best_pair]
                        i = 0
                        new_chars = []
                        while i < len(chars):
                            if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                                new_chars.append(new_token)
                                i += 2
                            else:
                                new_chars.append(chars[i])
                                i += 1
                        chars = new_chars
                        
                    for token in chars:
                        if token in self.vocab:
                            token_ids.append(self.vocab[token])
                        else:
                            if len(token) > 1:
                                for char in token:
                                    token_ids.append(self.vocab.get(char, self.unk_id))
                            else:
                                token_ids.append(self.vocab.get(token, self.unk_id))
                                
        return token_ids

    def decode(self, ids: List[int]) -> str:
        """
        Decodes list of token IDs back into string.
        """
        tokens = []
        for idx in ids:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                # Skip special tokens in normal text decoding
                if token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append("")
        return "".join(tokens)

    def save(self, filepath: str):
        """
        Saves the tokenizer vocabulary and merges list.
        """
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()}
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """
        Loads the tokenizer from file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found at {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        
        # Convert string keys back to tuples for merges
        self.merges = {}
        for k, v in data["merges"].items():
            parts = k.split("|||")
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = v

# Alias for backward compatibility with custom loaders
ByteTokenizer = SimpleBPETokenizer
