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

from tokenizers.core.base_tokenizer import (
    BaseTokenizer,
    TokenizedOutput,
    TokenizerOptions,
    TokenizerType,
)
from tokenizers.core.errors import TokenizationError
from tokenizers.encoding.encoder import Encoder
from tokenizers.encoding.special_tokens import SpecialTokens
from tokenizers.normalization.normalizers import Normalizer
from tokenizers.preprocessing.pipeline import PreprocessingPipeline
from tokenizers.utils.parallel import parallel_process


class WordpieceTokenizer(BaseTokenizer):
    """WordPiece tokenizer implementation."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        normalizer: Optional[Normalizer] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
        encoder: Optional[Encoder] = None,
        special_tokens: Optional[SpecialTokens] = None,
        options: Optional[TokenizerOptions] = None,
        wordpiece_prefix: str = "##",
        unk_token: str = "[UNK]",
        split_pattern: str = r"""'s|'t|'re|'ve|'m|'ll|'d|\s+|[^\w\s]+|\w+"""
    ):
        """Initialize a WordPiece tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary to train
            normalizer: Text normalizer component
            preprocessor: Text preprocessing pipeline
            encoder: Token encoder component
            special_tokens: Special tokens handler
            options: Tokenizer options
            wordpiece_prefix: Prefix for wordpiece subwords
            unk_token: Unknown token string
            split_pattern: Regex pattern for initial tokenization
        """
        super().__init__(
            tokenizer_type=TokenizerType.WORDPIECE,
            vocab_size=vocab_size,
            normalizer=normalizer,
            preprocessor=preprocessor,
            encoder=encoder,
            special_tokens=special_tokens,
            options=options,
        )
        
        # WordPiece-specific attributes
        self.wordpiece_prefix = wordpiece_prefix
        self.unk_token = unk_token
        self.split_pattern = split_pattern
        self._split_regex = re.compile(split_pattern)
        self.cache = {}
        
        self.logger.debug("Initialized WordPiece tokenizer")
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Perform initial tokenization of text."""
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        text = self.preprocessor.process(text)
        
        # Tokenize with regex
        return self._split_regex.findall(text)
    
    def _wordpiece_tokenize(self, word: str) -> List[str]:
        """Tokenize a word using WordPiece algorithm."""
        # Check cache first
        if word in self.cache:
            return self.cache[word]
        
        # If word is in vocabulary, return it as a single token
        if word in self.encoder.token_to_id_map:
            self.cache[word] = [word]
            return [word]
        
        # Try to split the word into subwords
        tokens = []
        start = 0
        max_end = len(word)
        
        # First try the whole word
        first_token = word
        if first_token in self.encoder.token_to_id_map:
            tokens.append(first_token)
            start = max_end
        else:
            first_token = word[:1]  # First character
            if first_token in self.encoder.token_to_id_map:
                tokens.append(first_token)
                start = 1
            else:
                # Unknown token
                self.cache[word] = [self.unk_token]
                return [self.unk_token]
        
        # Process the rest of the word
        while start < max_end:
            end = max_end
            cur_substr = None
            
            # Find the largest subword starting from current position
            while start < end:
                substr = (self.wordpiece_prefix + word[start:end]) if start > 0 else word[start:end]
                if substr in self.encoder.token_to_id_map:
                    cur_substr = substr
                    break
                end -= 1
            
            # If no valid subword found, word cannot be tokenized
            if cur_substr is None:
                tokens = [self.unk_token]
                break
            
            tokens.append(cur_substr)
            start = end
        
        # Cache result
        self.cache[word] = tokens
        return tokens
    
    def _encode_word(self, word: str) -> List[str]:
        """Encode a single word using WordPiece."""
        return self._wordpiece_tokenize(word)
    
    def train(self, texts: List[str], min_frequency: int = 2, num_workers: int = -1, **kwargs) -> None:
        """Train the WordPiece tokenizer on a corpus of texts.
        
        Args:
            texts: List of texts to train on
            min_frequency: Minimum frequency for a token to be included
            num_workers: Number of worker processes (-1 for all cores)
            **kwargs: Additional training parameters
        """
        if num_workers <= 0:
            num_workers = mp.cpu_count()
        
        self.logger.info(f"Training WordPiece tokenizer on {len(texts)} texts using {num_workers} workers")
        
        # Count word frequencies
        word_counts = defaultdict(int)
        
        # Pre-tokenize in parallel
        self.logger.debug("Pre-tokenizing texts")
        
        def process_text(text):
            tokens = self._pre_tokenize(text)
            for token in tokens:
                if token.strip():  # Skip empty tokens
                    word_counts[token] += 1
            return tokens
        
        all_tokens = []
        for i, tokens in enumerate(tqdm(parallel_process(texts, process_text, n_jobs=num_workers), total=len(texts))):
            all_tokens.extend(tokens)
        
        # Filter by frequency
        word_counts = {word: count for word, count in word_counts.items() if count >= min_frequency}
        
        # Initialize with basic vocabulary (single characters from all words)
        vocab = set()
        for word in word_counts:
            for char in word:
                vocab.add(char)
        
        # Add the special prefix for subwords
        if self.wordpiece_prefix not in vocab:
            vocab.add(self.wordpiece_prefix)
        
        # Add special tokens to vocabulary
        if self.special_tokens:
            for token in self.special_tokens.all_special_tokens:
                vocab.add(token)
        
        # WordPiece training algorithm
        self.logger.info(f"Initial vocab size: {len(vocab)}, target: {self.vocab_size}")
        
        # Generate potential subwords from all words
        potential_subwords = set()
        for word, count in word_counts.items():
            for i in range(1, len(word) + 1):
                for j in range(i + 1, len(word) + 1):
                    # Add prefix for non-initial subwords
                    subword = word[i:j]
                    if i > 0:
                        subword = self.wordpiece_prefix + subword
                    potential_subwords.add(subword)
        
        # Calculate scores for all potential subwords
        scores = {}
        for subword in tqdm(potential_subwords, desc="Scoring subwords"):
            score = 0
            for word, count in word_counts.items():
                # Count occurrences of subword in word
                if subword.startswith(self.wordpiece_prefix):
                    # Only match in middle or end of word
                    plain_subword = subword[len(self.wordpiece_prefix):]
                    pos = 1
                    while pos < len(word):
                        pos = word.find(plain_subword, pos)
                        if pos == -1:
                            break
                        score += count
                        pos += 1
                else:
                    # Match at start of word
                    if word.startswith(subword):
                        score += count
            
            scores[subword] = score
        
        # Sort subwords by score
        sorted_subwords = sorted(scores.items(), key=lambda x: -x[1])
        
        # Add top scoring subwords to vocabulary until we reach target size
        for subword, score in sorted_subwords:
            if len(vocab) >= self.vocab_size:
                break
            vocab.add(subword)
        
        # Ensure we have the unknown token
        if self.unk_token not in vocab:
            vocab.add(self.unk_token)
        
        # Create encoder with final vocabulary
        token_to_id = {token: i for i, token in enumerate(sorted(vocab))}
        self.encoder = Encoder(token_to_id)
        
        # Register special token IDs
        if self.special_tokens:
            for token_type, token in self.special_tokens.special_tokens.items():
                if token in self.encoder.token_to_id_map:
                    token_id = self.encoder.token_to_id_map[token]
                    self.special_tokens.register_special_token_id(token_type, token_id)
        
        self._is_trained = True
        self.cache = {}  # Reset cache
        
        self.logger.info(f"WordPiece training completed. Vocabulary size: {self.encoder.vocab_size}")
    
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
        text_tokens = []
        for word in self._pre_tokenize(text):
            if word.strip():  # Skip empty tokens
                text_tokens.extend(self._encode_word(word))
        
        text_ids = [self.token_to_id(token) for token in text_tokens]
        
        # Handle text pair if provided
        if text_pair:
            pair_tokens = []
            for word in self._pre_tokenize(text_pair):
                if word.strip():  # Skip empty tokens
                    pair_tokens.extend(self._encode_word(word))
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
                special_tokens_mask = None
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
        
                # Merge tokens and remove prefix from wordpieces
                text = ""
                for token in tokens:
                    if token.startswith(self.wordpiece_prefix):
                        text += token[len(self.wordpiece_prefix):]
                    else:
                        # Add space before tokens that don't have the prefix (except first token)
                        if text and not token.startswith(self.wordpiece_prefix):
                            text += " "
                        text += token
        
                return text
    
            def save(self, path: Union[str, Path], **kwargs) -> None:
                """Save the tokenizer to a directory.
        
                Args:
                    path: Directory path to save to
                    **kwargs: Additional saving parameters
                """
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
        
                # Save vocabulary
                if self.encoder:
                    vocab = self.encoder.get_vocab()
                    with open(path / "vocab.json", "w", encoding="utf-8") as f:
                        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
                # Save configuration
                config = {
                    "tokenizer_type": self.tokenizer_type.value,
                    "vocab_size": self.vocab_size,
                    "wordpiece_prefix": self.wordpiece_prefix,
                    "unk_token": self.unk_token,
                    "split_pattern": self.split_pattern,
                    "special_tokens": self.special_tokens.get_special_tokens_dict() if self.special_tokens else {},
                    "options": self.options.dict() if self.options else {},
                }
                with open(path / "config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
        
                self.logger.info(f"Saved WordPiece tokenizer to {path}")
    
            @classmethod
            def load(cls, path: Union[str, Path], **kwargs) -> "WordpieceTokenizer":
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
                    wordpiece_prefix=config.get("wordpiece_prefix", "##"),
                    unk_token=config.get("unk_token", "[UNK]"),
                    split_pattern=config.get("split_pattern", r"""'s|'t|'re|'ve|'m|'ll|'d|\s+|[^\w\s]+|\w+""")
                )
        
                # Register special token IDs
                if special_tokens:
                    for token_type, token in special_tokens.special_tokens.items():
                        if token in encoder.token_to_id_map:
                            token_id = encoder.token_to_id_map[token]
                            special_tokens.register_special_token_id(token_type, token_id)
        
                # Mark as trained
                tokenizer._is_trained = True
        
                logger.info(f"Loaded WordPiece tokenizer from {path}")
                return tokenizer            # Create token type