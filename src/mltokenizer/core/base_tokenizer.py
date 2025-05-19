from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from mltokenizer.core.errors import TokenizationError
from mltokenizer.encoding.encoder import Encoder
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import Normalizer
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline
from mltokenizer.utils.logging import tokenizer_logger


class TokenizerType(str, Enum):
    """Types of tokenizers supported by the system."""
    BPE = "bpe"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"
    SENTENCEPIECE = "sentencepiece"
    CHARACTER = "character"
    CUSTOM = "custom"


class TokenizerOptions(BaseModel):
    """Configuration options for tokenizers."""
    add_bos_token: bool = Field(default=False, description="Add beginning of sequence token")
    add_eos_token: bool = Field(default=False, description="Add end of sequence token")
    add_padding_token: bool = Field(default=False, description="Add padding token")
    truncation_strategy: str = Field(default="longest_first", 
                                     description="Strategy for truncation: longest_first, only_first, only_second")
    max_length: Optional[int] = Field(default=None, description="Maximum sequence length")
    stride: int = Field(default=0, description="Stride for truncation with overlap")
    padding_strategy: str = Field(default="longest", 
                                  description="Strategy for padding: longest, max_length, do_not_pad")
    pad_to_multiple_of: Optional[int] = Field(default=None, 
                                         description="Pad to a multiple of this value")
    return_attention_mask: bool = Field(default=True, description="Return attention mask")
    return_token_type_ids: bool = Field(default=True, description="Return token type IDs")
    return_overflowing_tokens: bool = Field(default=False, description="Return overflowing tokens")
    return_special_tokens_mask: bool = Field(default=False, description="Return special tokens mask")
    return_offsets_mapping: bool = Field(default=False, description="Return offsets mapping")
    return_length: bool = Field(default=False, description="Return sequence lengths")


class TokenizedOutput(BaseModel):
    """Standard output format for tokenization results."""
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    special_tokens_mask: Optional[List[int]] = None
    overflowing_tokens: Optional[List[List[int]]] = None
    offset_mapping: Optional[List[Tuple[int, int]]] = None
    length: Optional[int] = None
    tokens: Optional[List[str]] = None


class BatchTokenizedOutput(BaseModel):
    """Output format for batch tokenization."""
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None
    token_type_ids: Optional[List[List[int]]] = None
    special_tokens_mask: Optional[List[List[int]]] = None
    overflowing_tokens: Optional[List[List[List[int]]]] = None
    offset_mapping: Optional[List[List[Tuple[int, int]]]] = None
    lengths: Optional[List[int]] = None
    tokens: Optional[List[List[str]]] = None


class BaseTokenizer(ABC):
    """Base class for all tokenizers in the system."""
    
    def __init__(
        self,
        tokenizer_type: TokenizerType,
        vocab_size: int,
        normalizer: Optional[Normalizer] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
        encoder: Optional[Encoder] = None,
        special_tokens: Optional[SpecialTokens] = None,
        options: Optional[TokenizerOptions] = None,
    ):
        """Initialize the base tokenizer with components and options.
        
        Args:
            tokenizer_type: Type of tokenizer
            vocab_size: Size of the vocabulary
            normalizer: Text normalizer component
            preprocessor: Text preprocessing pipeline
            encoder: Token encoder component
            special_tokens: Special tokens handler
            options: Tokenizer options
        """
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.normalizer = normalizer
        self.preprocessor = preprocessor or PreprocessingPipeline()
        self.encoder = encoder
        self.special_tokens = special_tokens or SpecialTokens()
        self.options = options or TokenizerOptions()
        self._is_trained = False
        
        # Initialize logger
        self.logger = tokenizer_logger.bind(tokenizer_type=tokenizer_type.value)
        self.logger.info(f"Initialized {tokenizer_type.value} tokenizer with vocab size {vocab_size}")
    
    @property
    def is_trained(self) -> bool:
        """Check if the tokenizer is trained."""
        return self._is_trained
    
    @abstractmethod
    def train(self, texts: List[str], **kwargs) -> None:
        """Train the tokenizer on a corpus of texts.
        
        Args:
            texts: List of texts to train on
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def encode_batch(
        self,
        texts: List[str],
        text_pairs: Optional[List[str]] = None,
        add_special_tokens: bool = True,
        return_tokens: bool = False
    ) -> BatchTokenizedOutput:
        """Encode a batch of texts (or text pairs) into token IDs.
        
        Args:
            texts: List of texts to encode
            text_pairs: Optional list of paired texts
            add_special_tokens: Whether to add special tokens
            return_tokens: Whether to return the string tokens
            
        Returns:
            BatchTokenizedOutput object with tokenization results
        """
        self.logger.debug(f"Encoding batch of {len(texts)} texts")
        
        if not self.is_trained:
            raise TokenizationError("Tokenizer must be trained before encoding")
        
        # Validate input
        if text_pairs and len(texts) != len(text_pairs):
            raise ValueError("Number of texts and text pairs must match")
        
        # Process each text (or text pair)
        results = []
        for i, text in enumerate(texts):
            text_pair = text_pairs[i] if text_pairs else None
            result = self.encode(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                return_tokens=return_tokens
            )
            results.append(result)
        
        # Combine results into batch output
        batch_output = BatchTokenizedOutput(
            input_ids=[r.input_ids for r in results],
            attention_mask=[r.attention_mask for r in results] if results[0].attention_mask is not None else None,
            token_type_ids=[r.token_type_ids for r in results] if results[0].token_type_ids is not None else None,
            special_tokens_mask=[r.special_tokens_mask for r in results] if results[0].special_tokens_mask is not None else None,
            overflowing_tokens=[r.overflowing_tokens for r in results] if results[0].overflowing_tokens is not None else None,
            offset_mapping=[r.offset_mapping for r in results] if results[0].offset_mapping is not None else None,
            lengths=[r.length for r in results] if results[0].length is not None else None,
            tokens=[r.tokens for r in results] if return_tokens else None
        )
        
        return batch_output
    
    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token IDs back to texts.
        
        Args:
            batch_token_ids: List of token ID lists to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            List of decoded texts
        """
        self.logger.debug(f"Decoding batch of {len(batch_token_ids)} token sequences")
        
        if not self.is_trained:
            raise TokenizationError("Tokenizer must be trained before decoding")
        
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save the tokenizer to a directory.
        
        Args:
            path: Directory path to save to
            **kwargs: Additional saving parameters
        """
        raise NotImplementedError("Each tokenizer must implement its own save method")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "BaseTokenizer":
        """Load a tokenizer from a directory.
        
        Args:
            path: Directory path to load from
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded tokenizer instance
        """
        raise NotImplementedError("Each tokenizer must implement its own load method")
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        if not self.encoder:
            raise TokenizationError("Encoder is not initialized")
        
        return self.encoder.get_vocab()
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size.
        
        Returns:
            Size of the vocabulary
        """
        return self.vocab_size
    
    def id_to_token(self, id: int) -> str:
        """Convert a token ID to its string representation.
        
        Args:
            id: Token ID
            
        Returns:
            String token
        """
        if not self.encoder:
            raise TokenizationError("Encoder is not initialized")
        
        return self.encoder.id_to_token(id)
    
    def token_to_id(self, token: str) -> int:
        """Convert a string token to its ID.
        
        Args:
            token: String token
            
        Returns:
            Token ID
        """
        if not self.encoder:
            raise TokenizationError("Encoder is not initialized")
        
        return self.encoder.token_to_id(token)
    
    def add_tokens(self, tokens: List[str]) -> int:
        """Add new tokens to the vocabulary.
        
        Args:
            tokens: List of tokens to add
            
        Returns:
            Number of tokens actually added
        """
        if not self.encoder:
            raise TokenizationError("Encoder is not initialized")
        
        return self.encoder.add_tokens(tokens)
    
    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        """Add special tokens to the tokenizer.
        
        Args:
            special_tokens_dict: Dictionary of special tokens
            
        Returns:
            Number of tokens actually added
        """
        if not self.special_tokens:
            raise TokenizationError("Special tokens handler is not initialized")
        
        if not self.encoder:
            raise TokenizationError("Encoder is not initialized")
        
        # First register special tokens
        self.special_tokens.register_special_tokens(special_tokens_dict)
        
        # Then add them to the encoder
        added = 0
        for token_type, token in special_tokens_dict.items():
            if self.encoder.add_special_token(token, token_type):
                added += 1
        
        return added
    
    def apply_truncation(
        self, 
        input_ids: List[int],
        token_type_ids: Optional[List[int]] = None,
        pair_input_ids: Optional[List[int]] = None,
    ) -> Tuple[List[int], Optional[List[int]], Optional[List[List[int]]]]:
        """Apply truncation to input sequences according to the options.
        
        Args:
            input_ids: First sequence of token IDs
            token_type_ids: Token type IDs
            pair_input_ids: Optional second sequence of token IDs
            
        Returns:
            Tuple of (truncated input_ids, truncated token_type_ids, overflowing tokens)
        """
        if not self.options.max_length:
            return input_ids, token_type_ids, None
        
        max_length = self.options.max_length
        strategy = self.options.truncation_strategy
        stride = self.options.stride
        overflowing_tokens = []
        
        # Handle single sequence
        if pair_input_ids is None:
            if len(input_ids) <= max_length:
                return input_ids, token_type_ids, None
            
            # Simple truncation for single sequence
            overflowing_tokens = []
            for i in range(0, len(input_ids) - max_length + 1, max(1, stride)):
                overflowing_tokens.append(input_ids[i:i + max_length])
            
            return input_ids[:max_length], token_type_ids[:max_length] if token_type_ids else None, overflowing_tokens
        
        # Handle sequence pair
        total_len = len(input_ids) + len(pair_input_ids)
        if total_len <= max_length:
            return input_ids, token_type_ids, None
        
        # Apply different truncation strategies
        if strategy == "longest_first":
            # Truncate the longer sequence first
            if len(input_ids) > len(pair_input_ids):
                input_ids = input_ids[:max_length - len(pair_input_ids)]
            else:
                pair_input_ids = pair_input_ids[:max_length - len(input_ids)]
        elif strategy == "only_first":
            # Only truncate the first sequence
            input_ids = input_ids[:max_length - len(pair_input_ids)]
        elif strategy == "only_second":
            # Only truncate the second sequence
            pair_input_ids = pair_input_ids[:max_length - len(input_ids)]
        
        # Recombine input_ids and update token_type_ids
        combined_input_ids = input_ids + pair_input_ids
        combined_token_type_ids = None
        if token_type_ids:
            combined_token_type_ids = token_type_ids
        
        return combined_input_ids, combined_token_type_ids, overflowing_tokens
    
    def apply_padding(
        self,
        input_ids: List[int],
        max_length: Optional[int] = None,
        token_type_ids: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        """Apply padding to input sequence according to the options.
        
        Args:
            input_ids: Sequence of token IDs
            max_length: Maximum length to pad to (overrides options)
            token_type_ids: Token type IDs
            
        Returns:
            Tuple of (padded input_ids, attention_mask, padded token_type_ids)
        """
        if not self.special_tokens or not self.special_tokens.pad_token_id:
            raise TokenizationError("Padding token must be set to apply padding")
        
        length = len(input_ids)
        padding_strategy = self.options.padding_strategy
        pad_to_multiple_of = self.options.pad_to_multiple_of
        
        # Determine max length to pad to
        if max_length is None:
            if padding_strategy == "max_length" and self.options.max_length:
                max_length = self.options.max_length
            elif padding_strategy == "do_not_pad":
                return input_ids, [1] * length, token_type_ids
            else:  # padding_strategy == "longest"
                max_length = length
        
        # Adjust to multiple if needed
        if pad_to_multiple_of:
            max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        
        # No padding needed
        if length >= max_length:
            return input_ids, [1] * length, token_type_ids
        
        # Pad sequence
        padding_length = max_length - length
        padded_input_ids = input_ids + [self.special_tokens.pad_token_id] * padding_length
        attention_mask = [1] * length + [0] * padding_length
        
        # Pad token type IDs if provided
        padded_token_type_ids = None
        if token_type_ids:
            padded_token_type_ids = token_type_ids + [0] * padding_length
        
        return padded_input_ids, attention_mask, padded_token_type_ids
    
    def get_special_tokens_mask(
        self,
        input_ids: List[int],
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """Create a mask identifying special tokens in input_ids.
        
        Args:
            input_ids: Sequence of token IDs
            already_has_special_tokens: Whether input_ids already has special tokens
            
        Returns:
            List of 0s and 1s where 1 indicates a special token
        """
        if not self.special_tokens:
            return [0] * len(input_ids)
        
        if already_has_special_tokens:
            return [1 if id in self.special_tokens.all_special_ids else 0 for id in input_ids]
        
        # Need to add special tokens first
        raise NotImplementedError("Getting special tokens mask without adding special tokens is not supported yet") 