import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
from mltokenizer.core.base_tokenizer import TokenizedOutput
from mltokenizer.core.errors import TokenizationError, UntrainedTokenizerError
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import LowercaseNormalizer


@pytest.fixture
def sample_texts() -> List[str]:
    """Sample texts for training and testing."""
    return [
        "This is a test",
        "Another test sentence",
        "WordPiece tokenization works by splitting words into subwords",
        "The quick brown fox jumps over the lazy dog",
    ]


@pytest.fixture
def simple_vocab() -> Dict[str, int]:
    """A simple vocabulary for testing."""
    return {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "the": 5,
        "##s": 6,
        "is": 7,
        "a": 8,
        "test": 9,
        "word": 10,
        "piece": 11,
        "##ing": 12,
        "work": 13,
        "##s": 14,
        "by": 15,
        "split": 16,
        "##ing": 17,
        "in": 18,
        "##to": 19,
        "sub": 20,
        "##word": 21,
        "##s": 22,
    }


def test_wordpiece_tokenizer_initialization():
    """Test initialization of WordpieceTokenizer."""
    tokenizer = WordpieceTokenizer(vocab_size=100)
    assert tokenizer.tokenizer_type.value == "wordpiece"
    assert tokenizer.vocab_size == 100
    assert not tokenizer.is_trained


def test_wordpiece_tokenizer_train(sample_texts):
    """Test training a WordPiece tokenizer."""
    tokenizer = WordpieceTokenizer(vocab_size=50)
    tokenizer.train(sample_texts, min_frequency=1)
    
    assert tokenizer.is_trained
    assert tokenizer.vocab_size > 0
    assert len(tokenizer.get_vocab()) > 0


def test_wordpiece_tokenizer_with_vocab(simple_vocab):
    """Test initialization with a pre-defined vocabulary."""
    tokenizer = WordpieceTokenizer(
        vocab_size=len(simple_vocab),
        unk_token="[UNK]",
        wordpiece_prefix="##"
    )
    
    # Set the vocabulary manually
    tokenizer.encoder.token_to_id_map = simple_vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in simple_vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    for token_type, token_id in [
        ("pad_token", 0), ("unk_token", 1), ("cls_token", 2), 
        ("sep_token", 3), ("mask_token", 4)
    ]:
        tokenizer.special_tokens.register_special_token_id(token_type, token_id)
    
    # Test vocabulary size
    assert tokenizer.vocab_size == len(simple_vocab)
    assert tokenizer.is_trained


def test_wordpiece_tokenizer_encode(simple_vocab):
    """Test encoding text with WordPiece tokenizer."""
    tokenizer = WordpieceTokenizer(
        vocab_size=len(simple_vocab),
        unk_token="[UNK]",
        wordpiece_prefix="##"
    )
    
    # Set the vocabulary manually
    tokenizer.encoder.token_to_id_map = simple_vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in simple_vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    for token_type, token_id in [
        ("pad_token", 0), ("unk_token", 1), ("cls_token", 2), 
        ("sep_token", 3), ("mask_token", 4)
    ]:
        tokenizer.special_tokens.register_special_token_id(token_type, token_id)
    
    # Test encoding
    result = tokenizer.encode("this is a test", return_tokens=True)
    
    assert isinstance(result, TokenizedOutput)
    assert len(result.input_ids) > 0
    
    # Check if tokens are correctly identified
    if result.tokens:
        # Note: "this" might be broken down into "th" + "##is" or handled as UNK
        assert "is" in result.tokens
        assert "a" in result.tokens
        assert "test" in result.tokens


def test_wordpiece_tokenizer_decode(simple_vocab):
    """Test decoding token IDs with WordPiece tokenizer."""
    tokenizer = WordpieceTokenizer(
        vocab_size=len(simple_vocab),
        unk_token="[UNK]",
        wordpiece_prefix="##"
    )
    
    # Set the vocabulary manually
    tokenizer.encoder.token_to_id_map = simple_vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in simple_vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    for token_type, token_id in [
        ("pad_token", 0), ("unk_token", 1), ("cls_token", 2), 
        ("sep_token", 3), ("mask_token", 4)
    ]:
        tokenizer.special_tokens.register_special_token_id(token_type, token_id)
    
    # Test decoding a simple sequence
    tokens = [7, 8, 9]  # "is", "a", "test"
    decoded = tokenizer.decode(tokens)
    
    # WordPiece should join subwords, removing the prefix
    assert decoded.strip() == "is a test"


def test_wordpiece_tokenizer_save_load(sample_texts):
    """Test saving and loading a WordPiece tokenizer."""
    # Train a tokenizer
    tokenizer = WordpieceTokenizer(vocab_size=50)
    tokenizer.train(sample_texts, min_frequency=1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "wordpiece_tokenizer"
        
        # Save the tokenizer
        tokenizer.save(save_path)
        
        # Load the tokenizer
        loaded_tokenizer = WordpieceTokenizer.load(save_path)
        
        # Verify loaded tokenizer
        assert loaded_tokenizer.is_trained
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        assert len(loaded_tokenizer.get_vocab()) == len(tokenizer.get_vocab())


def test_wordpiece_tokenizer_with_normalizer(sample_texts):
    """Test WordPiece tokenizer with a normalizer."""
    normalizer = LowercaseNormalizer()
    tokenizer = WordpieceTokenizer(vocab_size=50, normalizer=normalizer)
    tokenizer.train(sample_texts, min_frequency=1)
    
    # Test case normalization
    result = tokenizer.encode("THIS IS A TEST", return_tokens=True)
    decoded = tokenizer.decode(result.input_ids)
    assert decoded.lower() == "this is a test"


def test_wordpiece_tokenizer_with_special_tokens(simple_vocab):
    """Test WordPiece tokenizer with special tokens."""
    special_tokens = SpecialTokens(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    tokenizer = WordpieceTokenizer(
        vocab_size=len(simple_vocab),
        special_tokens=special_tokens,
        wordpiece_prefix="##"
    )
    
    # Set the vocabulary manually
    tokenizer.encoder.token_to_id_map = simple_vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in simple_vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    for token_type, token_id in [
        ("pad_token", 0), ("unk_token", 1), ("cls_token", 2), 
        ("sep_token", 3), ("mask_token", 4)
    ]:
        tokenizer.special_tokens.register_special_token_id(token_type, token_id)
    
    # Test encoding with special tokens
    result = tokenizer.encode("this is a test", add_special_tokens=True, return_tokens=True)
    
    assert result.tokens is not None
    assert "[CLS]" in result.tokens
    assert "[SEP]" in result.tokens


def test_wordpiece_tokenizer_encode_pair(simple_vocab):
    """Test encoding a pair of texts."""
    special_tokens = SpecialTokens(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    tokenizer = WordpieceTokenizer(
        vocab_size=len(simple_vocab),
        special_tokens=special_tokens,
        wordpiece_prefix="##"
    )
    
    # Set the vocabulary manually
    tokenizer.encoder.token_to_id_map = simple_vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in simple_vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    for token_type, token_id in [
        ("pad_token", 0), ("unk_token", 1), ("cls_token", 2), 
        ("sep_token", 3), ("mask_token", 4)
    ]:
        tokenizer.special_tokens.register_special_token_id(token_type, token_id)
    
    # Test encoding a pair of texts
    result = tokenizer.encode(
        "this is", 
        text_pair="a test", 
        add_special_tokens=True, 
        return_tokens=True
    )
    
    assert result.tokens is not None
    assert result.tokens[0] == "[CLS]"
    assert "[SEP]" in result.tokens  # Should have at least one separator
    
    # Check token type IDs
    assert result.token_type_ids is not None
    # Token type IDs should indicate which segment each token belongs to (0 or 1)
    assert sum(result.token_type_ids) > 0  # At least some tokens should be from second segment


def test_wordpiece_tokenizer_subword_tokenization(simple_vocab):
    """Test subword tokenization mechanics."""
    tokenizer = WordpieceTokenizer(
        vocab_size=len(simple_vocab),
        unk_token="[UNK]",
        wordpiece_prefix="##"
    )
    
    # Set the vocabulary manually
    tokenizer.encoder.token_to_id_map = simple_vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in simple_vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    for token_type, token_id in [
        ("pad_token", 0), ("unk_token", 1), ("cls_token", 2), 
        ("sep_token", 3), ("mask_token", 4)
    ]:
        tokenizer.special_tokens.register_special_token_id(token_type, token_id)
    
    # Test encoding a word that should be split into subwords
    result = tokenizer.encode("working", return_tokens=True)
    
    assert result.tokens is not None
    # "working" should be split into "work" + "##ing"
    assert "work" in result.tokens
    assert "##ing" in result.tokens


def test_wordpiece_tokenizer_errors():
    """Test error handling in WordPiece tokenizer."""
    tokenizer = WordpieceTokenizer(vocab_size=100)
    
    # Test encoding before training
    with pytest.raises(UntrainedTokenizerError):
        tokenizer.encode("test")
    
    # Test decoding before training
    with pytest.raises(UntrainedTokenizerError):
        tokenizer.decode([1, 2, 3])
    
    # Test saving before training
    with pytest.raises(TokenizationError):
        tokenizer.save("/invalid/path")


if __name__ == "__main__":
    pytest.main() 