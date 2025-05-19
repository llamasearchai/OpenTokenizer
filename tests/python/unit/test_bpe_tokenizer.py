import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from mltokenizer.algorithms.bpe import BPETokenizer
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
        "BPE tokenization works by merging frequent pairs",
        "The quick brown fox jumps over the lazy dog",
    ]


def test_bpe_tokenizer_initialization():
    """Test initialization of BPETokenizer."""
    tokenizer = BPETokenizer(vocab_size=100)
    assert tokenizer.tokenizer_type.value == "bpe"
    assert tokenizer.vocab_size == 100
    assert not tokenizer.is_trained


def test_bpe_tokenizer_with_merges():
    """Test initialization with pre-defined merges."""
    merges = {("t", "h"): 0, ("th", "e"): 1}
    tokenizer = BPETokenizer(vocab_size=100, merges=merges)
    assert tokenizer.merges == merges


def test_bpe_tokenizer_train(sample_texts):
    """Test BPE tokenizer training."""
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(sample_texts, min_frequency=1)
    
    assert tokenizer.is_trained
    assert tokenizer.vocab_size > 0
    assert len(tokenizer.merges) > 0


def test_bpe_tokenizer_encode():
    """Test the encode method."""
    # Create a simple pre-trained tokenizer
    merges = {("t", "h"): 0, ("th", "e"): 1, ("a", "n"): 2, ("an", "d"): 3}
    vocab = {"t": 0, "h": 1, "e": 2, "th": 3, "the": 4, "a": 5, "n": 6, "d": 7, "an": 8, "and": 9}
    
    tokenizer = BPETokenizer(vocab_size=len(vocab))
    tokenizer.merges = merges
    tokenizer.encoder.token_to_id_map = vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in vocab.items()}
    tokenizer._is_trained = True
    
    # Test encoding
    result = tokenizer.encode("the and", return_tokens=True)
    assert isinstance(result, TokenizedOutput)
    assert len(result.input_ids) > 0
    if result.tokens:
        assert "the" in result.tokens
        assert "and" in result.tokens


def test_bpe_tokenizer_decode():
    """Test the decode method."""
    # Create a simple pre-trained tokenizer
    merges = {("t", "h"): 0, ("th", "e"): 1, ("a", "n"): 2, ("an", "d"): 3}
    vocab = {"t": 0, "h": 1, "e": 2, "th": 3, "the": 4, "a": 5, "n": 6, "d": 7, "an": 8, "and": 9}
    
    tokenizer = BPETokenizer(vocab_size=len(vocab))
    tokenizer.merges = merges
    tokenizer.encoder.token_to_id_map = vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in vocab.items()}
    tokenizer._is_trained = True
    
    # Test decoding
    tokens = [4, 9]  # "the", "and"
    decoded = tokenizer.decode(tokens)
    assert "the" in decoded
    assert "and" in decoded


def test_bpe_tokenizer_save_load(sample_texts):
    """Test saving and loading a BPE tokenizer."""
    # Train a tokenizer
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(sample_texts, min_frequency=1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bpe_tokenizer"
        
        # Save the tokenizer
        tokenizer.save(save_path)
        
        # Load the tokenizer
        loaded_tokenizer = BPETokenizer.load(save_path)
        
        # Verify loaded tokenizer
        assert loaded_tokenizer.is_trained
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        assert len(loaded_tokenizer.merges) == len(tokenizer.merges)


def test_bpe_tokenizer_with_normalizer(sample_texts):
    """Test BPE tokenizer with a normalizer."""
    normalizer = LowercaseNormalizer()
    tokenizer = BPETokenizer(vocab_size=50, normalizer=normalizer)
    tokenizer.train(sample_texts, min_frequency=1)
    
    # Test case normalization
    result = tokenizer.encode("THIS IS A TEST", return_tokens=True)
    decoded = tokenizer.decode(result.input_ids)
    assert decoded.lower() == "this is a test"


def test_bpe_tokenizer_with_special_tokens():
    """Test BPE tokenizer with special tokens."""
    special_tokens = SpecialTokens(
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        cls_token="[CLS]",
        sep_token="[SEP]"
    )
    
    tokenizer = BPETokenizer(vocab_size=100, special_tokens=special_tokens)
    
    # Set up a simple vocabulary and merges
    merges = {("t", "h"): 0, ("th", "e"): 1}
    vocab = {
        "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[CLS]": 4, "[SEP]": 5,
        "t": 6, "h": 7, "e": 8, "th": 9, "the": 10
    }
    
    tokenizer.merges = merges
    tokenizer.encoder.token_to_id_map = vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    for token_type, token_id in [
        ("pad_token", 0), ("unk_token", 1), ("bos_token", 2), 
        ("eos_token", 3), ("cls_token", 4), ("sep_token", 5)
    ]:
        tokenizer.special_tokens.register_special_token_id(token_type, token_id)
    
    # Test encoding with special tokens
    result = tokenizer.encode("the", add_special_tokens=True, return_tokens=True)
    assert result.tokens is not None
    assert "[CLS]" in result.tokens
    assert "the" in result.tokens
    assert "[SEP]" in result.tokens


def test_bpe_tokenizer_encode_pair():
    """Test encoding a pair of texts."""
    # Create a simple pre-trained tokenizer
    merges = {("t", "h"): 0, ("th", "e"): 1, ("a", "n"): 2, ("an", "d"): 3}
    vocab = {
        "[CLS]": 0, "[SEP]": 1,
        "t": 2, "h": 3, "e": 4, "th": 5, "the": 6, 
        "a": 7, "n": 8, "d": 9, "an": 10, "and": 11
    }
    
    special_tokens = SpecialTokens(
        cls_token="[CLS]",
        sep_token="[SEP]"
    )
    
    tokenizer = BPETokenizer(vocab_size=len(vocab), special_tokens=special_tokens)
    tokenizer.merges = merges
    tokenizer.encoder.token_to_id_map = vocab
    tokenizer.encoder.id_to_token_map = {v: k for k, v in vocab.items()}
    tokenizer._is_trained = True
    
    # Register special token IDs
    tokenizer.special_tokens.register_special_token_id("cls_token", 0)
    tokenizer.special_tokens.register_special_token_id("sep_token", 1)
    
    # Test encoding a pair of texts
    result = tokenizer.encode(
        "the", 
        text_pair="and", 
        add_special_tokens=True, 
        return_tokens=True
    )
    
    assert result.tokens is not None
    assert result.tokens[0] == "[CLS]"
    assert "the" in result.tokens
    assert "[SEP]" in result.tokens
    assert "and" in result.tokens
    
    # Check token type IDs
    assert result.token_type_ids is not None
    # Token type IDs should indicate which segment each token belongs to
    # 0 for first segment, 1 for second segment
    assert sum(result.token_type_ids) > 0  # At least some tokens should be from second segment


def test_bpe_tokenizer_errors():
    """Test error handling in BPE tokenizer."""
    tokenizer = BPETokenizer(vocab_size=100)
    
    # Test encoding before training
    with pytest.raises(TokenizationError):
        tokenizer.encode("test")
    
    # Test decoding before training
    with pytest.raises(TokenizationError):
        tokenizer.decode([1, 2, 3])
    
    # Test saving before training
    with pytest.raises(TokenizationError):
        tokenizer.save("/invalid/path")


if __name__ == "__main__":
    pytest.main() 