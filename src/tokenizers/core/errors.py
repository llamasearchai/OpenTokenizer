from typing import Optional


class TokenizationError(Exception):
    """Base exception for tokenization errors."""
    pass


class PreprocessingError(TokenizationError):
    """Exception raised during text preprocessing."""
    pass


class EncodingError(TokenizationError):
    """Exception raised during token encoding."""
    pass


class DecodingError(TokenizationError):
    """Exception raised during token decoding."""
    pass


class VocabularyError(TokenizationError):
    """Exception raised for vocabulary-related issues."""
    pass


class TokenizerNotTrainedError(TokenizationError):
    """Exception raised when attempting to use an untrained tokenizer."""
    pass


class InvalidConfigurationError(TokenizationError):
    """Exception raised for invalid tokenizer configuration."""
    pass


class SerializationError(TokenizationError):
    """Exception raised during tokenizer serialization/deserialization."""
    pass