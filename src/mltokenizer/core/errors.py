class TokenizationError(Exception):
    """Base exception for tokenization errors."""
    pass

class UntrainedTokenizerError(TokenizationError):
    """Error raised when an untrained tokenizer is used."""
    def __init__(self, message="Tokenizer is not trained yet."):
        super().__init__(message)

class VocabularyError(TokenizationError):
    """Error related to vocabulary issues."""
    pass

# Add other specific error types as needed 