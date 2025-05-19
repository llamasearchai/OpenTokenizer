# src/mltokenizer/algorithms/bpe.py
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Set, Tuple, Union
import multiprocessing as mp

import numpy as np
from loguru import logger
from tqdm import tqdm

from mltokenizer.core.base_tokenizer import (
    BaseTokenizer,
    TokenizedOutput,
    TokenizerOptions,
    TokenizerType,
)
from mltokenizer.core.errors import TokenizationError
from mltokenizer.encoding.encoder import Encoder
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import Normalizer
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline
from mltokenizer.utils.parallel import parallel_process


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer implementation."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        normalizer: Optional[Normalizer] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
        encoder: Optional[Encoder] = None,
        special_tokens: Optional[SpecialTokens] = None,
        options: Optional[TokenizerOptions] = None,
        merges: Optional[Dict[Tuple[str, str], int]] = None,
        pre_tokenizer_pattern: Optional[str] = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    ):
        """Initialize a BPE tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary to train
            normalizer: Text normalizer component
            preprocessor: Text preprocessing pipeline
            encoder: Token encoder component
            special_tokens: Special tokens handler
            options: Tokenizer options
            merges: Pre-initialized BPE merges
            pre_tokenizer_pattern: Regex pattern for initial tokenization
        """
        super().__init__(
            tokenizer_type=TokenizerType.BPE,
            vocab_size=vocab_size,
            normalizer=normalizer,
            preprocessor=preprocessor,
            encoder=encoder,
            special_tokens=special_tokens,
            options=options,
        )
        
        # BPE-specific attributes
        self.merges = merges or {}
        self.pre_tokenizer_pattern = pre_tokenizer_pattern
        self._pre_tokenizer_regex = re.compile(pre_tokenizer_pattern)
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        
        self.logger.debug("Initialized BPE tokenizer")
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """Mapping from bytes to unicode characters for byte-level BPE."""
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs.copy()
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Perform initial tokenization of text."""
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        text = self.preprocessor.process(text)
        
        # Tokenize with regex
        tokens = self._pre_tokenizer_regex.findall(text)
        
        # Convert to byte-level representation
        bpe_tokens = []
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_chars = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_tokens.append(token_chars)
        
        return bpe_tokens
    
    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _merge_word(self, word: List[str]) -> List[str]:
        """Apply BPE merges to a single word."""
        # Handle caching
        word_str = "".join(word)
        if word_str in self.cache:
            return self.cache[word_str]
        
        pairs = self._get_pairs(word)
        if not pairs:
            return word
        
        while True:
            # Find the best pair to merge
            best_pair = None
            best_rank = float("inf")
            
            for pair in pairs:
                if pair in self.merges and self.merges[pair] < best_rank:
                    best_pair = pair
                    best_rank = self.merges[pair]
            
            if best_pair is None:
                break
                
            # Apply the merge
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            if len(word) == 1:
                break
            
            pairs = self._get_pairs(word)
        
        # Cache and return
        self.cache[word_str] = word
        return word
    
    def _bpe_encode(self, text: str) -> List[str]:
        """Encode text using BPE algorithm."""
        tokens = []
        for word in self._pre_tokenize(text):
            word = list(word)
            merged = self._merge_word(word)
            tokens.extend(merged)
        return tokens
    
    def train(self, texts: List[str], min_frequency: int = 2, num_workers: int = -1, **kwargs) -> None:
        """Train the BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of texts to train on
            min_frequency: Minimum frequency for a pair to be considered for merging
            num_workers: Number of worker processes (-1 for all cores)
            **kwargs: Additional training parameters
        """
        if num_workers <= 0:
            num_workers = mp.cpu_count()
        
        self.logger.info(f"Training BPE tokenizer on {len(texts)} texts using {num_workers} workers")
        
        # Initial vocabulary (characters)
        vocab: Dict[str, int] = defaultdict(int)
        
        # Count word frequencies
        word_freqs: Dict[str, int] = defaultdict(int)
        
        # Pre-tokenize in parallel
        self.logger.debug("Pre-tokenizing texts")
        
        def process_text(text):
            result = []
            for word in self._pre_tokenize(text):
                word_freqs[word] += 1
                for char in word:
                    vocab[char] += 1
                result.append(word)
            return result
        
        all_words = []
        for i, words in enumerate(tqdm(parallel_process(texts, process_text, n_jobs=num_workers), total=len(texts))):
            all_words.extend(words)
        
        # Filter by frequency
        word_freqs = {k: v for k, v in word_freqs.items() if v >= min_frequency}
        
        # Split words into character sequences
        splits = {word: list(word) for word in word_freqs.keys()}
        
        # Track merges
        merges: Dict[Tuple[str, str], int] = {}
        vocab_size = len(vocab)
        
        self.logger.info(f"Initial vocab size: {vocab_size}, target: {self.vocab_size}")
        
        # Main BPE training loop
        with tqdm(total=self.vocab_size - vocab_size) as pbar:
            while vocab_size < self.vocab_size:
                # Count pair frequencies
                pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
                for word, freq in word_freqs.items():
                    word_splits = splits[word]
                    for i in range(len(word_splits) - 1):
                        pair = (word_splits[i], word_splits[i + 1])
                        pair_freqs[pair] += freq
                
                if not pair_freqs:
                    break
                
                # Find most frequent pair
                best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
                
                # Record the merge operation
                merges[best_pair] = len(merges)
                
                # Apply the merge to all splits
                new_splits = {}
                for word in word_freqs:
                    word_splits = splits[word]
                    new_word = []
                    i = 0
                    while i < len(word_splits):
                        if i < len(word_splits) - 1 and (word_splits[i], word_splits[i + 1]) == best_pair:
                            new_word.append(word_splits[i] + word_splits[i + 1])
                            i += 2
                        else:
                            new_word.append(word_splits[i])
                            i += 1
                    new_splits[word] = new_word
                
                # Update the vocabulary
                new_token = best_pair[0] + best_pair[1]
                vocab[new_token] = pair_freqs[best_pair]
                vocab_size += 1
                
                # Update splits
                splits = new_splits
                
                pbar.update(1)
                pbar.set_description(f"Merging {best_pair[0]}+{best_pair[1]}")
        
        # Store the trained model
        self.merges = merges
        
        # Build encoder vocabulary
        token_to_id = {token: i for i, token in enumerate(vocab.keys())}
        
        # Create the encoder
        self.encoder = Encoder(token_to_id)
        
        # Add special tokens if needed
        if self.special_tokens:
            self.add_special_tokens(self.special_tokens.get_special_tokens_dict())
        
        self._is_trained = True
        self.cache = {}  # Reset cache
        
        self.logger.info(f"BPE training completed. Vocabulary size: {self.encoder.vocab_size}")
    
    def encode(
        self, 
        text: str, 
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        return_tokens: bool = False
    ) -> TokenizedOutput:
        """Encode a text (or text pair) into token IDs.
        
        Args:
            text: Text to encode
            text_pair: Optional paired text (e.g., for sequence-pair tasks)
            add_special_tokens: Whether to add special tokens
            return_tokens: Whether to return the string tokens
            
        Returns:
            TokenizedOutput object with tokenization results
        """
        if not self.is_trained:
            raise TokenizationError("Tokenizer must be trained before encoding")
        
        # Preprocess and tokenize
        text_tokens = self._bpe_encode(text)
        text_ids = [self.token_to_id(token) for token in text_tokens]
        
        # Handle text pair if provided
        if text_pair:
            pair_tokens = self._bpe_encode(text_pair)
            pair_ids = [self.token_to_id(token) for token in pair_tokens]
            
            # Create token type IDs (0 for first sequence, 1 for second)
            token_type_ids = [0] * len(text_ids) + [1] * len(pair_ids)
            input_ids = text_ids + pair_ids
            
            # Track tokens if requested
            tokens = text_tokens + pair_tokens if return_tokens else None
        else:
            token_type_ids = [0] * len(text_ids)
            input_ids = text_ids
            tokens = text_tokens if return_tokens else None
        
        # Add special tokens if requested
        if add_special_tokens and self.special_tokens:
            if text_pair:
                # Add format: [CLS] X [SEP] Y [SEP]
                cls_id = self.special_tokens.cls_token_id
                sep_id = self.special_tokens.sep_token_id
                
                if cls_id is not None and sep_id is not None:
                    special_input_ids = [cls_id] + text_ids + [sep_id] + pair_ids + [sep_id]
                    special_token_type_ids = [0] + token_type_ids[:len(text_ids)] + [0] + token_type_ids[len(text_ids):] + [1]
                    special_tokens_mask = [1] + [0] * len(text_ids) + [1] + [0] * len(pair_ids) + [1]
                    
                    # Update
                    input_ids = special_input_ids
                    token_type_ids = special_token_type_ids
                    
                    # Update tokens if tracking
                    if tokens:
                        cls_token = self.id_to_token(cls_id)
                        sep_token = self.id_to_token(sep_id)
                        tokens = [cls_token] + text_tokens + [sep_token] + pair_tokens + [sep_token]
            else:
                # Add format: [CLS] X [SEP]
                cls_id = self.special_tokens.cls_token_id
                sep_id = self.special_tokens.sep_token_id
                
                if cls_id is not None and sep_id is not None:
                    special_input_ids = [cls_id] + text_ids + [sep_id]
                    special_token_type_ids = [0] + token_type_ids + [0]
                    special_tokens_mask = [1] + [0] * len(text_ids) + [1]
                    
                    # Update
                    input_ids = special_input_ids
                    token_type_ids = special_token_type_ids
                    
                    # Update tokens if tracking
                    if tokens:
                        cls_token = self.id_to_token(cls_id)
                        sep_token = self.id_to_token(sep_id)
                        tokens = [cls_token] + text_tokens + [sep_token]
        else:
            special_tokens_mask = [0] * len(input_ids)
        
        # Apply truncation if needed
        input_ids, token_type_ids, overflowing_tokens = self.apply_truncation(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        # Apply padding if needed
        input_ids, attention_mask, token_type_ids = self.apply_padding(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        # Handle return options
        result = TokenizedOutput(
            input_ids=input_ids,
            attention_mask=attention_mask if self.options.return_attention_mask else None,
            token_type_ids=token_type_ids if self.options.return_token_type_ids else None,
            special_tokens_mask=special_tokens_mask if self.options.return_special_tokens_mask else None,
            overflowing_tokens=overflowing_tokens if self.options.return_overflowing_tokens else None,
            length=len(input_ids) if self.options.return_length else None,
            tokens=tokens
        )
        
        return result
    
    def decode(
        self, 
        token_ids: List[int], 
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not self.is_trained:
            raise TokenizationError("Tokenizer must be trained before decoding")
        
        # Filter special tokens if requested
        if skip_special_tokens and self.special_tokens:
            token_ids = [id for id in token_ids if id not in self.special_tokens.all_special_ids]
        
        # Convert IDs to tokens
        tokens = [self.id_to_token(id) for id in token_ids]
        
        # Merge tokens (handling byte-level encoding)
        text = "".join(tokens)
        
        # Convert from byte-level encoding back to UTF-8
        bytes_tokens = []
        for char in text:
            if char in self.byte_decoder:
                bytes_tokens.append(self.byte_decoder[char])
        
        # Decode bytes to text
        try:
            text = bytes(bytes_tokens).decode("utf-8", errors="replace")
        except Exception as e:
            self.logger.warning(f"Error decoding bytes to UTF-8: {e}")
            text = "".join(chr(b) for b in bytes_tokens)
        
        return text
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save the tokenizer to a directory.
        
        Args:
            path: Directory path to save to
            **kwargs: Additional saving parameters
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save merges
        with open(path / "merges.json", "w", encoding="utf-8") as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            json.dump(serializable_merges, f, ensure_ascii=False, indent=2)
        
        # Save vocabulary
        if self.encoder:
            vocab = self.encoder.get_vocab()
            with open(path / "vocab.json", "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        # Save configuration
        config = {
            "tokenizer_type": self.tokenizer_type.value,
            "vocab_size": self.vocab_size,
            "pre_tokenizer_pattern": self.pre_tokenizer_pattern,
            "special_tokens": self.special_tokens.get_special_tokens_dict() if self.special_tokens else {},
            "options": self.options.dict() if self.options else {},
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved BPE tokenizer to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "BPETokenizer":
        """Load a tokenizer from a directory.
        
        Args:
            path: Directory path to load from
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded tokenizer instance
        """
        path = Path(path)
        
        # Load configuration
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Load vocabulary
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        
        # Load merges
        with open(path / "merges.json", "r", encoding="utf-8") as f:
            serialized_merges = json.load(f)
            # Convert string keys back to tuples
            merges = {tuple(k.split(",")): v for k, v in serialized_merges.items()}
        
        # Create special tokens
        special_tokens = SpecialTokens()
        special_tokens.register_special_tokens(config.get("special_tokens", {}))
        
        # Create options
        options = TokenizerOptions(**config.get("options", {}))
        
        # Create encoder
        encoder = Encoder(vocab)
        
        # Create the tokenizer
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            encoder=encoder,
            special_tokens=special_tokens,
            options=options,
            merges=merges,
            pre_tokenizer_pattern=config.get("pre_tokenizer_pattern", r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # Mark as trained
        tokenizer._is_trained = True
        
        logger.info(f"Loaded BPE tokenizer from {path}")
        return tokenizer 