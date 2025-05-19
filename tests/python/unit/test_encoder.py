import pytest

from mltokenizer.encoding.encoder import Encoder


def test_encoder_initialization():
    """Test encoder initialization with vocabulary."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    assert encoder.vocab_size == 4
    assert encoder.token_to_id("hello") == 2
    assert encoder.id_to_token(3) == "world"


def test_encoder_get_vocab():
    """Test getting the vocabulary."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    vocab = encoder.get_vocab()
    assert vocab == token_to_id
    assert vocab is not token_to_id  # Should be a copy, not the same object


def test_encoder_token_to_id():
    """Test converting tokens to IDs."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    assert encoder.token_to_id("hello") == 2
    assert encoder.token_to_id("world") == 3
    
    # Unknown tokens should return UNK token ID (1)
    assert encoder.token_to_id("unknown") == 1


def test_encoder_id_to_token():
    """Test converting IDs to tokens."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    assert encoder.id_to_token(2) == "hello"
    assert encoder.id_to_token(3) == "world"
    
    # Unknown IDs should return UNK token
    assert encoder.id_to_token(999) == "[UNK]"


def test_encoder_encode():
    """Test encoding token sequences."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    tokens = ["hello", "world", "unknown"]
    ids = encoder.encode(tokens)
    
    assert ids == [2, 3, 1]  # "hello", "world", "[UNK]"


def test_encoder_decode():
    """Test decoding ID sequences."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    ids = [2, 3, 1, 999]  # "hello", "world", "[UNK]", unknown ID
    tokens = encoder.decode(ids)
    
    assert tokens == ["hello", "world", "[UNK]", "[UNK]"]


def test_encoder_add_tokens():
    """Test adding new tokens to the vocabulary."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    # Add new tokens
    added = encoder.add_tokens(["foo", "bar", "hello"])  # "hello" already exists
    
    assert added == 2  # Only 2 new tokens should be added
    assert encoder.vocab_size == 6
    assert encoder.token_to_id("foo") == 4
    assert encoder.token_to_id("bar") == 5
    assert encoder.token_to_id("hello") == 2  # Existing token ID should not change


def test_encoder_add_special_token():
    """Test adding special tokens to the vocabulary."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    # Add a new special token
    added = encoder.add_special_token("[SEP]", "sep_token")
    
    assert added is True
    assert encoder.vocab_size == 5
    assert encoder.token_to_id("[SEP]") == 4
    assert encoder.is_special_token("[SEP]")
    
    # Add an existing token as special
    added = encoder.add_special_token("hello", "greeting_token")
    
    assert added is False  # Should return False as token already exists
    assert encoder.vocab_size == 5  # Size should not change
    assert encoder.is_special_token("hello")


def test_encoder_is_special_token():
    """Test checking if a token is special."""
    token_to_id = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encoder = Encoder(token_to_id)
    
    # Mark a token as special
    encoder.add_special_token("[PAD]", "pad_token")
    encoder.add_special_token("[UNK]", "unk_token")
    
    assert encoder.is_special_token("[PAD]")
    assert encoder.is_special_token("[UNK]")
    assert not encoder.is_special_token("hello")
    assert not encoder.is_special_token("world")


if __name__ == "__main__":
    pytest.main() 