import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from tokenizers.normalization.unicode import UnicodeNormalizer


class Normalizer(ABC):
    """Base class for text normalizers."""
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize a text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        pass


class ComposeNormalizer(Normalizer):
    """Composite normalizer that applies multiple normalizers in sequence."""
    
    def __init__(self, normalizers: List[Normalizer]):
        """Initialize with a list of normalizers.
        
        Args:
            normalizers: List of normalizers to apply in sequence
        """
        self.normalizers = normalizers
    
    def normalize(self, text: str) -> str:
        """Apply all normalizers in sequence.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        for normalizer in self.normalizers:
            text = normalizer.normalize(text)
        return text


class LowercaseNormalizer(Normalizer):
    """Normalizer that converts text to lowercase."""
    
    def normalize(self, text: str) -> str:
        """Convert text to lowercase.
        
        Args:
            text: Text to normalize
            
        Returns:
            Lowercase text
        """
        return text.lower()


class StripNormalizer(Normalizer):
    """Normalizer that strips whitespace."""
    
    def __init__(self, strip_left: bool = True, strip_right: bool = True):
        """Initialize with stripping options.
        
        Args:
            strip_left: Whether to strip leading whitespace
            strip_right: Whether to strip trailing whitespace
        """
        self.strip_left = strip_left
        self.strip_right = strip_right
    
    def normalize(self, text: str) -> str:
        """Strip whitespace from text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Stripped text
        """
        if self.strip_left and self.strip_right:
            return text.strip()
        elif self.strip_left:
            return text.lstrip()
        elif self.strip_right:
            return text.rstrip()
        return text


class RegexNormalizer(Normalizer):
    """Normalizer that applies regex replacements."""
    
    def __init__(self, patterns_and_replacements: List[tuple]):
        """Initialize with patterns and replacements.
        
        Args:
            patterns_and_replacements: List of (pattern, replacement) tuples
        """
        self.patterns = []
        for pattern, replacement in patterns_and_replacements:
            if isinstance(pattern, str):
                pattern = re.compile(pattern)
            self.patterns.append((pattern, replacement))
    
    def normalize(self, text: str) -> str:
        """Apply regex replacements to text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text


class WhitespaceNormalizer(Normalizer):
    """Normalizer that normalizes whitespace."""
    
    def normalize(self, text: str) -> str:
        """Normalize whitespace in text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text


class SequenceNormalizer(Normalizer):
    """Factory class for creating common normalizer sequences."""
    
    @staticmethod
    def default() -> ComposeNormalizer:
        """Create a default normalizer sequence.
        
        Returns:
            Composite normalizer with common normalization steps
        """
        return ComposeNormalizer([
            UnicodeNormalizer(),
            LowercaseNormalizer(),
            StripNormalizer(),
            WhitespaceNormalizer()
        ])
    
    @staticmethod
    def bert_normalizer() -> ComposeNormalizer:
        """Create a normalizer sequence for BERT-like models.
        
        Returns:
            Composite normalizer for BERT-like models
        """
        return ComposeNormalizer([
            UnicodeNormalizer(form="NFC"),
            StripNormalizer(),
            WhitespaceNormalizer()
        ])
    
    @staticmethod
    def gpt_normalizer() -> ComposeNormalizer:
        """Create a normalizer sequence for GPT-like models.
        
        Returns:
            Composite normalizer for GPT-like models
        """
        return ComposeNormalizer([
            UnicodeNormalizer(form="NFC"),
            RegexNormalizer([
                # Fix broken UTF-8 characters
                (r'â', "'"),
                (r'â', '"'),
                (r'â', '"'),
                # Remove control characters
                (r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ''),
            ]),
            StripNormalizer(),
            WhitespaceNormalizer()
        ])