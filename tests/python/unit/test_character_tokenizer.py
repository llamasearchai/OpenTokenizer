import json
from pathlib import Path
import tempfile
import pytest

from mltokenizer.algorithms.character import CharacterTokenizer
from mltokenizer.core.errors import UntrainedTokenizerError, TokenizationError
from mltokenizer.core.base_tokenizer import TokenizedOutput
from mltokenizer.encoding.special_tokens import SpecialTokens


def test_character_tokenizer_initialization():
    """Test initialization of CharacterTokenizer with and without pre-defined vocab."""
    # Test with no pre-defined vocab
    tokenizer = CharacterTokenizer()
    assert not tokenizer.is_trained
    assert tokenizer.vocab_size == 0

    # Test with pre-defined vocab
    vocab = ["a", "b", "c"]
    tokenizer = CharacterTokenizer(char_vocab_list=vocab)
    assert tokenizer.is_trained
    assert tokenizer.vocab_size == len(vocab) + len(tokenizer.special_tokens.all_special_tokens)


def test_character_tokenizer_train():
    """Test the training method of CharacterTokenizer."""
    tokenizer = CharacterTokenizer()
    texts = ["hello", "world"]
    tokenizer.train(texts)
    assert tokenizer.is_trained
    assert tokenizer.vocab_size > 0


def test_character_tokenizer_encode():
    """Test the encode method of CharacterTokenizer."""
    vocab = ["h", "e", "l", "o", "w", "r", "d"]
    tokenizer = CharacterTokenizer(char_vocab_list=vocab)
    output = tokenizer.encode("hello")
    assert isinstance(output, TokenizedOutput)
    assert len(output.input_ids) > 0


def test_character_tokenizer_decode():
    """Test the decode method of CharacterTokenizer."""
    vocab = ["h", "e", "l", "o"]
    tokenizer = CharacterTokenizer(char_vocab_list=vocab)
    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded.input_ids)
    assert decoded == "hello"


def test_character_tokenizer_save_and_load():
    """Test the save and load methods of CharacterTokenizer."""
    vocab = ["a", "b", "c"]
    tokenizer = CharacterTokenizer(char_vocab_list=vocab)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tokenizer"
        tokenizer.save(save_path)
        loaded_tokenizer = CharacterTokenizer.load(save_path)
        assert loaded_tokenizer.is_trained
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


def test_character_tokenizer_errors():
    """Test error handling in CharacterTokenizer."""
    tokenizer = CharacterTokenizer()
    with pytest.raises(UntrainedTokenizerError):
        tokenizer.encode("test")
    with pytest.raises(UntrainedTokenizerError):
        tokenizer.decode([1, 2, 3])
    with pytest.raises(TokenizationError):
        tokenizer.save("/invalid/path")


if __name__ == "__main__":
    pytest.main() 