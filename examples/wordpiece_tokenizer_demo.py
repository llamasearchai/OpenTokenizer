#!/usr/bin/env python
"""
WordPiece Tokenizer Demo

This script demonstrates how to use the WordPiece Tokenizer from MLTokenizer,
configured in the style of BERT tokenizers. It shows training, encoding, decoding,
and handling of subword units.
"""

from pathlib import Path
import json

from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
from mltokenizer.normalization.normalizers import ComposeNormalizer, StripNormalizer, WhitespaceNormalizer
from mltokenizer.encoding.special_tokens import SpecialTokens


def main():
    """Run the WordPiece Tokenizer demo."""
    print("WordPiece Tokenizer Demo (BERT-style)")
    print("====================================\n")

    # Create components - note that BERT typically doesn't lowercase by default
    normalizer = ComposeNormalizer([
        StripNormalizer(),
        WhitespaceNormalizer()
    ])
    
    # Create BERT-style special tokens
    special_tokens = SpecialTokens(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = WordpieceTokenizer(
        vocab_size=200,  # Small vocabulary for quick demonstration
        normalizer=normalizer,
        special_tokens=special_tokens,
        wordpiece_prefix="##"  # BERT-style subword prefix
    )

    # Create training data
    print("Preparing training data...")
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "WordPiece tokenization is used in BERT and related models.",
        "It splits words into subwords using a greedy algorithm.",
        "Words are broken into smaller pieces when they're not in the vocabulary.",
        "Subword pieces are marked with a prefix like ##.",
        "For example, 'playing' might become 'play' and '##ing'.",
        "This helps the model handle unseen words by recognizing familiar parts.",
        "WordPiece is particularly effective for languages with rich morphology.",
        "It provides a good balance between character-level and word-level tokenization.",
        "In BERT, the tokenizer starts with '[CLS]' and separates sequences with '[SEP]'.",
        "The special tokens have important roles in the model architecture.",
        "This demonstration shows how WordPiece works in practice.",
        "Unlike BPE, WordPiece uses a different algorithm for creating subwords.",
        "It starts with a base vocabulary and adds the most useful subwords.",
        "Usefulness is determined by how much adding a subword increases likelihood.",
    ]

    # Train tokenizer
    print("Training WordPiece tokenizer (this may take a moment)...")
    tokenizer.train(train_texts)
    print(f"Trained vocabulary size: {tokenizer.vocab_size}")

    # Show part of the vocabulary
    vocab = tokenizer.get_vocab()
    print("\nSample from vocabulary:")
    # Sort by ID for cleaner output
    vocab_items = sorted(vocab.items(), key=lambda x: x[1])[:20]
    for token, id in vocab_items:
        # Replace special characters for display
        display_token = token.replace("\n", "\\n").replace("\t", "\\t")
        print(f"  '{display_token}': {id}")
    
    # Show which subword pieces were learned
    print("\nSubword pieces in vocabulary:")
    subword_pieces = [token for token in vocab.keys() if token.startswith("##")][:15]
    for piece in subword_pieces:
        print(f"  '{piece}'")

    # Encode a simple text
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"\nEncoding: '{test_text}'")
    
    # Encode with BERT-style special tokens
    encoded = tokenizer.encode(test_text, add_special_tokens=True, return_tokens=True)
    print("\nBERT-style encoding (with special tokens):")
    print(f"Input IDs: {encoded.input_ids}")
    if encoded.tokens:
        # Print tokens with their IDs for better visualization
        print("Tokens with IDs:")
        for i, (token, token_id) in enumerate(zip(encoded.tokens, encoded.input_ids)):
            print(f"  {i}: {token} ({token_id})")
    
    # Decode back to text
    decoded = tokenizer.decode(encoded.input_ids)
    print(f"\nDecoded text: '{decoded}'")
    
    # Decode without special tokens
    decoded_no_special = tokenizer.decode(encoded.input_ids, skip_special_tokens=True)
    print(f"Decoded (skip special tokens): '{decoded_no_special}'")

    # Demonstrate handling of unknown words
    complex_text = "Supercalifragilisticexpialidocious is antidisestablishmentarianism."
    print(f"\nHandling complex words: '{complex_text}'")
    
    complex_result = tokenizer.encode(complex_text, return_tokens=True)
    if complex_result.tokens:
        print("Tokens:")
        for token in complex_result.tokens:
            print(f"  '{token}'")
    
    # Demonstrate BERT-style sequence pair encoding
    text_a = "How are you?"
    text_b = "I am fine, thank you!"
    print(f"\nEncoding text pair (BERT-style): '{text_a}' + '{text_b}'")
    
    pair_result = tokenizer.encode(text_a, text_pair=text_b, add_special_tokens=True, return_tokens=True)
    
    print("\nText pair encoding:")
    if pair_result.tokens and pair_result.token_type_ids:
        print("Tokens with segment IDs:")
        for i, (token, type_id) in enumerate(zip(pair_result.tokens, pair_result.token_type_ids)):
            segment = "A" if type_id == 0 else "B"
            print(f"  {i}: {token} (Segment {segment})")
    
    # Save and load the tokenizer
    save_path = Path("./wordpiece_tokenizer_test")
    print(f"\nSaving tokenizer to {save_path}")
    tokenizer.save(save_path)
    
    print(f"Loading tokenizer from {save_path}")
    loaded_tokenizer = WordpieceTokenizer.load(save_path)
    
    # Verify the loaded tokenizer
    encoded_loaded = loaded_tokenizer.encode(test_text, add_special_tokens=True)
    print("\nVerifying loaded tokenizer:")
    print(f"Match: {encoded.input_ids == encoded_loaded.input_ids}")
    
    # Demonstrate mask token usage (for MLM tasks)
    masked_text = "The [MASK] brown fox jumps over the lazy dog."
    print(f"\nUsing mask token: '{masked_text}'")
    
    masked_result = tokenizer.encode(masked_text, add_special_tokens=True, return_tokens=True)
    print("Tokens with IDs:")
    if masked_result.tokens:
        for i, (token, token_id) in enumerate(zip(masked_result.tokens, masked_result.input_ids)):
            print(f"  {i}: {token} ({token_id})")
    
    # Check if [MASK] token was properly recognized
    if masked_result.tokens:
        mask_indices = [i for i, token in enumerate(masked_result.tokens) if token == "[MASK]"]
        print(f"[MASK] token found at positions: {mask_indices}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 