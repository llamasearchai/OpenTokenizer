import tempfile
from pathlib import Path

import pytest

from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
from mltokenizer.algorithms.character import CharacterTokenizer
from mltokenizer.core.base_tokenizer import TokenizerType
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import (
    ComposeNormalizer, 
    LowercaseNormalizer, 
    StripNormalizer,
    WhitespaceNormalizer
)
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline
from mltokenizer.preprocessing.text_cleaning import (
    RemoveHTMLTagsProcessor,
    RemoveURLsProcessor,
    RemoveEmailsProcessor
)


@pytest.fixture
def sample_texts():
    """Sample texts for training and testing."""
    return [
        "This is a test",
        "Another test sentence",
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        "Tokenization is the process of breaking text into smaller pieces called tokens",
        "Machine learning models process text as numerical data"
    ]


def test_bpe_with_normalizer_and_preprocessing():
    """Test BPE tokenizer with normalizer and preprocessing pipeline."""
    # Create a normalizer
    normalizer = ComposeNormalizer([
        LowercaseNormalizer(),
        StripNormalizer(),
        WhitespaceNormalizer()
    ])
    
    # Create a preprocessing pipeline
    preprocessor = PreprocessingPipeline([
        RemoveHTMLTagsProcessor(),
        RemoveURLsProcessor(),
        RemoveEmailsProcessor()
    ])
    
    # Create special tokens
    special_tokens = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>"
    )
    
    # Create and train the tokenizer
    tokenizer = BPETokenizer(
        vocab_size=100,
        normalizer=normalizer,
        preprocessor=preprocessor,
        special_tokens=special_tokens
    )
    
    # Train on simple texts
    train_texts = [
        "This is a test",
        "Another test",
        "The quick brown fox",
        "Lorem ipsum dolor sit amet"
    ]
    tokenizer.train(train_texts, min_frequency=1)
    
    # Test with clean text
    clean_text = "This is a simple test sentence"
    result = tokenizer.encode(clean_text, return_tokens=True)
    
    # Test with text that needs preprocessing
    html_text = "<p>This is a <b>test</b> with HTML</p>"
    result_html = tokenizer.encode(html_text, return_tokens=True)
    
    # Text with URL and email
    complex_text = "Contact us at info@example.com or visit https://example.com"
    result_complex = tokenizer.encode(complex_text, return_tokens=True)
    
    # Verify results
    assert tokenizer.is_trained
    assert result.input_ids is not None
    assert len(result.input_ids) > 0
    
    # Verify preprocessing worked
    if result_html.tokens:
        assert "<p>" not in result_html.tokens
        assert "<b>" not in result_html.tokens
    
    if result_complex.tokens:
        assert "info@example.com" not in result_complex.tokens
        assert "https://example.com" not in result_complex.tokens
    
    # Test round-trip encoding and decoding
    encoded = tokenizer.encode("This is a test")
    decoded = tokenizer.decode(encoded.input_ids)
    assert decoded.lower() == "this is a test"  # May be lowercase due to normalizer


def test_wordpiece_specialization_for_bert():
    """Test WordPiece tokenizer configured for BERT-like models."""
    # Create BERT-style special tokens
    special_tokens = SpecialTokens(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    # Create a minimal normalizer (BERT typically doesn't lowercase)
    normalizer = ComposeNormalizer([
        StripNormalizer(),
        WhitespaceNormalizer()
    ])
    
    # Create the tokenizer
    tokenizer = WordpieceTokenizer(
        vocab_size=30522,  # BERT vocab size
        normalizer=normalizer,
        special_tokens=special_tokens,
        wordpiece_prefix="##"
    )
    
    # Train on sample texts
    train_texts = [
        "This is a test",
        "Another test sentence with some longer words",
        "WordPiece tokenization works by splitting words into subwords",
        "The quick brown fox jumps over the lazy dog"
    ]
    tokenizer.train(train_texts, min_frequency=1)
    
    # Test BERT-style encoding with special tokens and text pairs
    text_a = "This is a test"
    text_b = "This is a pair"
    
    result = tokenizer.encode(
        text_a, 
        text_pair=text_b,
        add_special_tokens=True,
        return_tokens=True
    )
    
    # Verify BERT-style formatting
    assert result.tokens is not None
    assert result.tokens[0] == "[CLS]"
    assert "[SEP]" in result.tokens
    assert result.token_type_ids is not None
    assert sum(result.token_type_ids) > 0  # Should have some tokens from the second segment
    
    # Test decoding with special tokens
    decoded_with_special = tokenizer.decode(result.input_ids, skip_special_tokens=False)
    decoded_without_special = tokenizer.decode(result.input_ids, skip_special_tokens=True)
    
    assert "[CLS]" in decoded_with_special
    assert "[SEP]" in decoded_with_special
    assert "[CLS]" not in decoded_without_special
    assert "[SEP]" not in decoded_without_special


def test_tokenizer_save_load_compatibility():
    """Test saving and loading different tokenizer types maintain compatibility."""
    tokenizers = [
        BPETokenizer(vocab_size=100),
        WordpieceTokenizer(vocab_size=100),
        CharacterTokenizer()
    ]
    
    # Train each tokenizer
    train_texts = [
        "This is a test",
        "Another test",
        "The quick brown fox",
        "Lorem ipsum dolor sit amet"
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = []
        
        # Train, encode, save, load, and encode again with each tokenizer
        for i, tokenizer in enumerate(tokenizers):
            tokenizer.train(train_texts, min_frequency=1)
            
            # Initial encoding
            original_output = tokenizer.encode("This is a test", return_tokens=True)
            
            # Save the tokenizer
            save_path = Path(tmpdir) / f"tokenizer_{i}"
            tokenizer.save(save_path)
            
            # Load the tokenizer
            if tokenizer.tokenizer_type == TokenizerType.BPE:
                loaded_tokenizer = BPETokenizer.load(save_path)
            elif tokenizer.tokenizer_type == TokenizerType.WORDPIECE:
                loaded_tokenizer = WordpieceTokenizer.load(save_path)
            elif tokenizer.tokenizer_type == TokenizerType.CHARACTER:
                loaded_tokenizer = CharacterTokenizer.load(save_path)
            
            # Encode with loaded tokenizer
            loaded_output = loaded_tokenizer.encode("This is a test", return_tokens=True)
            
            # Verify results are identical
            assert original_output.input_ids == loaded_output.input_ids
            assert original_output.tokens == loaded_output.tokens
            
            results.append((tokenizer.tokenizer_type.value, loaded_tokenizer.tokenizer_type.value))
        
        # Verify all tokenizer types loaded correctly
        assert ('bpe', 'bpe') in results
        assert ('wordpiece', 'wordpiece') in results
        assert ('character', 'character') in results


def test_end_to_end_tokenization_pipeline():
    """Test a complete end-to-end tokenization pipeline."""
    # Create components
    normalizer = ComposeNormalizer([
        LowercaseNormalizer(),
        StripNormalizer(),
        WhitespaceNormalizer()
    ])
    
    preprocessor = PreprocessingPipeline([
        RemoveHTMLTagsProcessor(),
        RemoveURLsProcessor()
    ])
    
    special_tokens = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>"
    )
    
    # Create tokenizer
    tokenizer = BPETokenizer(
        vocab_size=100,
        normalizer=normalizer,
        preprocessor=preprocessor,
        special_tokens=special_tokens
    )
    
    # Sample text with preprocessing needs
    raw_text = """
    <html>
    <body>
        <p>Visit our website at https://example.com</p>
        <p>This is a sample text with HTML tags.</p>
    </body>
    </html>
    """
    
    # Expected clean text (approximately)
    expected_clean = "visit our website at this is a sample text with html tags"
    
    # Train the tokenizer on clean text
    train_texts = [
        "This is a sample text",
        "Visit our website",
        "Sample text with tags",
        "HTML is a markup language"
    ]
    tokenizer.train(train_texts, min_frequency=1)
    
    # Process the text through the entire pipeline
    result = tokenizer.encode(raw_text, add_special_tokens=True, return_tokens=True)
    decoded = tokenizer.decode(result.input_ids, skip_special_tokens=True)
    
    # Check results
    assert tokenizer.is_trained
    assert result.input_ids is not None
    assert len(result.input_ids) > 0
    
    # Check that special tokens were added
    special_token_ids = [
        tokenizer.special_tokens.bos_token_id,
        tokenizer.special_tokens.eos_token_id
    ]
    if special_token_ids[0] is not None:
        assert special_token_ids[0] in result.input_ids
    if special_token_ids[1] is not None:
        assert special_token_ids[1] in result.input_ids
    
    # Check that preprocessing was applied
    assert "<html>" not in decoded
    assert "<body>" not in decoded
    assert "<p>" not in decoded
    assert "https://example.com" not in decoded
    
    # Check the normalized output is similar to expected (may not be exact due to BPE tokenization)
    assert "visit" in decoded.lower()
    assert "website" in decoded.lower()
    assert "sample" in decoded.lower()
    assert "text" in decoded.lower()


if __name__ == "__main__":
    pytest.main() 