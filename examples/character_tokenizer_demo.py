#!/usr/bin/env python
"""
Character Tokenizer Demo

This script demonstrates how to use the CharacterTokenizer from MLTokenizer.
It shows training, encoding, decoding, and saving/loading functionality.
"""

from pathlib import Path
import json

from mltokenizer.algorithms.character import CharacterTokenizer
from mltokenizer.normalization.normalizers import ComposeNormalizer, LowercaseNormalizer, StripNormalizer
from mltokenizer.encoding.special_tokens import SpecialTokens


def main():
    """Run the CharacterTokenizer demo."""
    print("Character Tokenizer Demo")
    print("========================\n")

    # Create a normalizer
    normalizer = ComposeNormalizer([
        LowercaseNormalizer(),
        StripNormalizer()
    ])

    # Create special tokens
    special_tokens = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>"
    )

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = CharacterTokenizer(
        normalizer=normalizer,
        special_tokens=special_tokens
    )

    # Create training data
    train_texts = [
        "Hello, world!",
        "This is a test of the character tokenizer.",
        "It breaks text into individual characters.",
        "Special tokens can be added at the beginning and end.",
        "Character tokenization is simple but effective for certain tasks.",
    ]

    # Train tokenizer
    print("Training tokenizer...")
    tokenizer.train(train_texts)
    print(f"Trained vocabulary size: {tokenizer.vocab_size}")

    # Show the vocabulary
    vocab = tokenizer.get_vocab()
    print("\nVocabulary:")
    print(json.dumps(vocab, indent=2)[:500] + "..." if len(json.dumps(vocab, indent=2)) > 500 else json.dumps(vocab, indent=2))

    # Encode a text
    test_text = "Hello, world! This is a test."
    print(f"\nEncoding: '{test_text}'")
    
    # Encode without special tokens
    encoded_regular = tokenizer.encode(test_text, return_tokens=True)
    print("\nRegular encoding (without special tokens):")
    print(f"Input IDs: {encoded_regular.input_ids}")
    if encoded_regular.tokens:
        print(f"Tokens: {encoded_regular.tokens}")
    
    # Encode with special tokens
    encoded_special = tokenizer.encode(test_text, add_special_tokens=True, return_tokens=True)
    print("\nEncoding with special tokens:")
    print(f"Input IDs: {encoded_special.input_ids}")
    if encoded_special.tokens:
        print(f"Tokens: {encoded_special.tokens}")
    
    # Decode back to text
    decoded_text = tokenizer.decode(encoded_special.input_ids)
    print(f"\nDecoded text: '{decoded_text}'")
    
    # Decode without special tokens
    decoded_no_special = tokenizer.decode(encoded_special.input_ids, skip_special_tokens=True)
    print(f"Decoded (skip special tokens): '{decoded_no_special}'")
    
    # Save and load the tokenizer
    save_path = Path("./character_tokenizer_test")
    print(f"\nSaving tokenizer to {save_path}")
    tokenizer.save(save_path)
    
    print(f"Loading tokenizer from {save_path}")
    loaded_tokenizer = CharacterTokenizer.load(save_path)
    
    # Verify the loaded tokenizer
    encoded_loaded = loaded_tokenizer.encode(test_text, add_special_tokens=True)
    print("\nVerifying loaded tokenizer:")
    print(f"Original IDs: {encoded_special.input_ids}")
    print(f"Loaded IDs:   {encoded_loaded.input_ids}")
    print(f"Match: {encoded_special.input_ids == encoded_loaded.input_ids}")
    
    # Try with a text pair
    text_a = "First sequence"
    text_b = "Second sequence"
    print(f"\nEncoding text pair: '{text_a}' + '{text_b}'")
    
    pair_result = tokenizer.encode(text_a, text_pair=text_b, add_special_tokens=True, return_tokens=True)
    print("\nText pair encoding:")
    print(f"Input IDs: {pair_result.input_ids}")
    if pair_result.tokens:
        print(f"Tokens: {pair_result.tokens}")
    if pair_result.token_type_ids:
        print(f"Token types: {pair_result.token_type_ids}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 