import pytest

from mltokenizer.encoding.special_tokens import SpecialTokens


def test_special_tokens_initialization():
    """Test initialization with default values."""
    st = SpecialTokens()
    
    # Check default values
    assert st.pad_token == "[PAD]"
    assert st.unk_token == "[UNK]"
    assert st.cls_token == "[CLS]"
    assert st.sep_token == "[SEP]"
    assert st.mask_token == "[MASK]"
    assert st.bos_token is None
    assert st.eos_token is None


def test_special_tokens_custom_initialization():
    """Test initialization with custom values."""
    st = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=["<special1>", "<special2>"]
    )
    
    # Check custom values
    assert st.pad_token == "<pad>"
    assert st.unk_token == "<unk>"
    assert st.cls_token == "<cls>"
    assert st.sep_token == "<sep>"
    assert st.mask_token == "<mask>"
    assert st.bos_token == "<bos>"
    assert st.eos_token == "<eos>"
    
    # Check additional tokens in all_special_tokens
    all_tokens = st.all_special_tokens
    assert "<special1>" in all_tokens
    assert "<special2>" in all_tokens


def test_register_special_token():
    """Test registering a special token."""
    st = SpecialTokens()
    
    # Register a new token type
    st.register_special_token("custom_token", "<custom>")
    
    # Check if the token is accessible via the special_tokens dictionary
    assert st.special_tokens.get("custom_token") == "<custom>"


def test_register_special_token_id():
    """Test registering a special token ID."""
    st = SpecialTokens()
    
    # Register token and its ID
    st.register_special_token("custom_token", "<custom>")
    st.register_special_token_id("custom_token", 42)
    
    # Check if the ID is accessible
    assert st.special_token_ids.get("custom_token") == 42


def test_register_special_tokens():
    """Test registering multiple special tokens."""
    st = SpecialTokens()
    
    # Register multiple tokens
    tokens_dict = {
        "custom_token1": "<custom1>",
        "custom_token2": "<custom2>",
    }
    st.register_special_tokens(tokens_dict)
    
    # Check if all tokens are registered
    assert st.special_tokens.get("custom_token1") == "<custom1>"
    assert st.special_tokens.get("custom_token2") == "<custom2>"


def test_get_special_tokens_dict():
    """Test getting the special tokens dictionary."""
    st = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<cls>"
    )
    
    # Get the dictionary
    tokens_dict = st.get_special_tokens_dict()
    
    # Check the dictionary content
    assert tokens_dict.get("pad_token") == "<pad>"
    assert tokens_dict.get("unk_token") == "<unk>"
    assert tokens_dict.get("cls_token") == "<cls>"
    
    # Verify it's a copy, not the original
    assert tokens_dict is not st.special_tokens


def test_token_properties():
    """Test token property getters."""
    st = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        bos_token="<bos>",
        eos_token="<eos>"
    )
    
    # Check property getters
    assert st.pad_token == "<pad>"
    assert st.unk_token == "<unk>"
    assert st.cls_token == "<cls>"
    assert st.sep_token == "<sep>"
    assert st.mask_token == "<mask>"
    assert st.bos_token == "<bos>"
    assert st.eos_token == "<eos>"


def test_token_id_properties():
    """Test token ID property getters."""
    st = SpecialTokens()
    
    # Register token IDs
    st.register_special_token_id("pad_token", 0)
    st.register_special_token_id("unk_token", 1)
    st.register_special_token_id("cls_token", 2)
    
    # Check property getters
    assert st.pad_token_id == 0
    assert st.unk_token_id == 1
    assert st.cls_token_id == 2
    assert st.sep_token_id is None  # Not registered yet
    
    # Register the remaining tokens
    st.register_special_token_id("sep_token", 3)
    st.register_special_token_id("mask_token", 4)
    st.register_special_token_id("bos_token", 5)
    st.register_special_token_id("eos_token", 6)
    
    # Check property getters again
    assert st.sep_token_id == 3
    assert st.mask_token_id == 4
    assert st.bos_token_id == 5
    assert st.eos_token_id == 6


def test_all_special_tokens():
    """Test getting all special tokens."""
    st = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<cls>",
        sep_token="<sep>",
        additional_special_tokens=["<special1>", "<special2>"]
    )
    
    # Get all special tokens
    all_tokens = st.all_special_tokens
    
    # Check if all tokens are included
    assert "<pad>" in all_tokens
    assert "<unk>" in all_tokens
    assert "<cls>" in all_tokens
    assert "<sep>" in all_tokens
    assert "<special1>" in all_tokens
    assert "<special2>" in all_tokens
    
    # Check if duplicates are handled correctly
    st.register_special_token("duplicate_token", "<pad>")  # Duplicate of pad_token
    all_tokens = st.all_special_tokens
    assert all_tokens.count("<pad>") == 1  # Should appear only once


def test_all_special_ids():
    """Test getting all special token IDs."""
    st = SpecialTokens()
    
    # Register token IDs
    st.register_special_token_id("pad_token", 0)
    st.register_special_token_id("unk_token", 1)
    st.register_special_token_id("cls_token", 2)
    
    # Get all special token IDs
    all_ids = st.all_special_ids
    
    # Check if all IDs are included
    assert 0 in all_ids
    assert 1 in all_ids
    assert 2 in all_ids
    
    # Check if the length matches
    assert len(all_ids) == 3


if __name__ == "__main__":
    pytest.main() 