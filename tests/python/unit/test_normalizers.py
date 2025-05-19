import re
import pytest

from mltokenizer.normalization.normalizers import (
    Normalizer,
    ComposeNormalizer,
    LowercaseNormalizer,
    StripNormalizer,
    RegexNormalizer,
    WhitespaceNormalizer,
    SequenceNormalizer
)
from mltokenizer.normalization.unicode import UnicodeNormalizer


def test_lowercase_normalizer():
    """Test LowercaseNormalizer."""
    normalizer = LowercaseNormalizer()
    
    # Test with mixed case
    assert normalizer.normalize("Hello World") == "hello world"
    assert normalizer.normalize("UPPERCASE") == "uppercase"
    assert normalizer.normalize("lowercase") == "lowercase"
    assert normalizer.normalize("MiXeD CaSe") == "mixed case"
    
    # Test with non-alphabetical characters
    assert normalizer.normalize("Hello123!") == "hello123!"
    assert normalizer.normalize("SYMBOLS: @#$%") == "symbols: @#$%"


def test_strip_normalizer():
    """Test StripNormalizer."""
    # Test with default settings (strip both sides)
    normalizer = StripNormalizer()
    assert normalizer.normalize("  Hello  ") == "Hello"
    assert normalizer.normalize("\t\nHello\t\n") == "Hello"
    
    # Test with left stripping only
    normalizer = StripNormalizer(strip_left=True, strip_right=False)
    assert normalizer.normalize("  Hello  ") == "Hello  "
    
    # Test with right stripping only
    normalizer = StripNormalizer(strip_left=False, strip_right=True)
    assert normalizer.normalize("  Hello  ") == "  Hello"
    
    # Test with no stripping
    normalizer = StripNormalizer(strip_left=False, strip_right=False)
    assert normalizer.normalize("  Hello  ") == "  Hello  "


def test_regex_normalizer():
    """Test RegexNormalizer."""
    # Define patterns and replacements
    patterns_and_replacements = [
        (r'\d+', 'NUM'),  # Replace digits with NUM
        (r'[^\w\s]', ''),  # Remove punctuation
        (re.compile(r'\s+'), ' ')  # Normalize whitespace (using compiled regex)
    ]
    
    normalizer = RegexNormalizer(patterns_and_replacements)
    
    # Test basic replacements
    assert normalizer.normalize("Hello123!") == "HelloNUM"
    assert normalizer.normalize("price: $9.99") == "price NUM"
    assert normalizer.normalize("multiple    spaces") == "multiple spaces"
    
    # Test with complex patterns
    assert normalizer.normalize("Hello, my zip is 12345!") == "Hello my zip is NUM"


def test_whitespace_normalizer():
    """Test WhitespaceNormalizer."""
    normalizer = WhitespaceNormalizer()
    
    # Test with various whitespace patterns
    assert normalizer.normalize("Hello  World") == "Hello World"
    assert normalizer.normalize("Multiple    Spaces") == "Multiple Spaces"
    assert normalizer.normalize("Tabs\tand\tSpaces") == "Tabs and Spaces"
    assert normalizer.normalize("Newlines\nAre\nNormalized") == "Newlines Are Normalized"
    assert normalizer.normalize(" Leading and trailing spaces ") == " Leading and trailing spaces "  # Doesn't strip


def test_unicode_normalizer():
    """Test UnicodeNormalizer."""
    # Test with default form (NFC)
    normalizer = UnicodeNormalizer()
    
    # Test NFC normalization
    assert normalizer.normalize("café") == "café"  # Should be normalized to NFC form
    
    # Test with NFKC form (compatibility decomposition followed by composition)
    normalizer = UnicodeNormalizer(form="NFKC")
    
    # NFKC should normalize special characters like typographic quotes
    fancy_quotes = "\u201Chello\u201D"  # Unicode fancy quotes
    normalized = normalizer.normalize(fancy_quotes)
    assert normalized != fancy_quotes  # Should normalize to standard quotes
    
    # Test with NFD form (canonical decomposition)
    normalizer = UnicodeNormalizer(form="NFD")
    
    # NFD should decompose characters
    nfd_result = normalizer.normalize("café")
    assert len(nfd_result) > len("café")  # NFD expands characters


def test_compose_normalizer():
    """Test ComposeNormalizer."""
    # Create a composition of normalizers
    normalizers = [
        StripNormalizer(),
        LowercaseNormalizer(),
        WhitespaceNormalizer()
    ]
    
    normalizer = ComposeNormalizer(normalizers)
    
    # Test sequential application
    assert normalizer.normalize("  HELLO  WORLD  ") == "hello world"
    
    # Add a regex normalizer to the composition
    regex_normalizer = RegexNormalizer([(r'\d+', 'NUM')])
    normalizers.append(regex_normalizer)
    
    normalizer = ComposeNormalizer(normalizers)
    assert normalizer.normalize("  HELLO 123  ") == "hello NUM"


def test_sequence_normalizer_default():
    """Test SequenceNormalizer.default()."""
    normalizer = SequenceNormalizer.default()
    
    # Default normalizer should apply unicode normalization, lowercasing, 
    # stripping, and whitespace normalization
    assert normalizer.normalize("  HELLO  WORLD  ") == "hello world"
    assert normalizer.normalize("Café") == "café"


def test_sequence_normalizer_bert():
    """Test SequenceNormalizer.bert_normalizer()."""
    normalizer = SequenceNormalizer.bert_normalizer()
    
    # BERT normalizer should apply unicode normalization, 
    # stripping, and whitespace normalization, but not lowercasing
    result = normalizer.normalize("  HELLO  WORLD  ")
    assert result == "HELLO WORLD"  # Should preserve case


def test_sequence_normalizer_gpt():
    """Test SequenceNormalizer.gpt_normalizer()."""
    normalizer = SequenceNormalizer.gpt_normalizer()
    
    # GPT normalizer should apply unicode normalization, 
    # special character fixes, stripping, and whitespace normalization
    result = normalizer.normalize("  Hello  World  ")
    assert result == "Hello World"
    
    # Test with broken UTF-8 characters if possible
    # This is a simplified test as actual broken UTF-8 is hard to reproduce
    result = normalizer.normalize("Hello â World")  # â is a placeholder for broken character
    assert "â" not in result or result == "Hello ' World"  # Should fix or remove the character


if __name__ == "__main__":
    pytest.main() 