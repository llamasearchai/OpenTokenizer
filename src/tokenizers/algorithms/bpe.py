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