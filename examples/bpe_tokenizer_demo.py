#!/usr/bin/env python
"""
BPE Tokenizer Demo

This script demonstrates how to use the BPE (Byte-Pair Encoding) Tokenizer from MLTokenizer.
It shows training, encoding, decoding, and saving/loading functionality.
"""

from pathlib import Path
import json

from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.normalization.normalizers import SequenceNormalizer
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline
from mltokenizer.preprocessing.text_cleaning import RemoveHTMLTagsProcessor
from mltokenizer.encoding.special_tokens import SpecialTokens


def main():
    """Run the BPE Tokenizer demo."""
    print("BPE Tokenizer Demo")
    print("=================\n")

    # Create components
    normalizer = SequenceNormalizer.default()  # This includes Unicode normalization, lowercasing, etc.
    
    preprocessor = PreprocessingPipeline([
        RemoveHTMLTagsProcessor()
    ])
    
    special_tokens = SpecialTokens(
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>"
    )

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = BPETokenizer(
        vocab_size=100,  # Small vocabulary for quick demonstration
        normalizer=normalizer,
        preprocessor=preprocessor,
        special_tokens=special_tokens
    )

    # Create training data - for BPE we want more text to learn meaningful merges
    print("Preparing training data...")
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "BPE tokenization works by iteratively merging the most frequent pairs of characters or tokens.",
        "This creates a vocabulary of subword units that efficiently represent the training corpus.",
        "It handles unseen words better than word-level tokenization by breaking them into known subwords.",
        "The algorithm starts with characters and builds up to larger units.",
        "When the vocabulary size is reached, the merging process stops.",
        "BPE is used in many modern language models like GPT and RoBERTa.",
        "This tokenizer implements the algorithm with various optimizations.",
        "You can configure the vocabulary size to balance specificity and generality.",
        "Smaller vocabularies mean more token IDs used per text, but better handling of unseen words.",
        "Larger vocabularies are more efficient for common words but may struggle with rare words.",
        "This is a demonstration of BPE training and usage in MLTokenizer.",
        "The quick brown fox jumps over the lazy dog and runs far away.",
        "The early bird catches the worm but the second mouse gets the cheese.",
        "All that glitters is not gold; all who wander are not lost.",
    ]

    # Train tokenizer
    print("Training BPE tokenizer (this may take a moment)...")
    tokenizer.train(train_texts, min_frequency=2)
    print(f"Trained vocabulary size: {tokenizer.vocab_size}")

    # Display some of the learned merges
    print("\nSome learned BPE merges:")
    merge_items = list(tokenizer.merges.items())[:10]
    for pair, rank in merge_items:
        print(f"  {pair[0]} + {pair[1]} → {pair[0] + pair[1]} (rank: {rank})")

    # Show part of the vocabulary
    vocab = tokenizer.get_vocab()
    print("\nSample from vocabulary:")
    vocab_items = list(vocab.items())[:20]
    for token, id in vocab_items:
        # Replace newlines and tabs for display
        display_token = token.replace("\n", "\\n").replace("\t", "\\t")
        print(f"  '{display_token}': {id}")

    # Encode a text
    test_text = "The quick brown fox jumps over the lazy dog."
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

    # Try with a text containing unseen words
    unseen_text = "Supercalifragilisticexpialidocious is a made-up word."
    print(f"\nEncoding text with unseen words: '{unseen_text}'")
    
    unseen_result = tokenizer.encode(unseen_text, return_tokens=True)
    print("Tokenization of unseen text:")
    print(f"Input IDs: {unseen_result.input_ids}")
    if unseen_result.tokens:
        print(f"Tokens: {unseen_result.tokens}")
    
    # Save and load the tokenizer
    save_path = Path("./bpe_tokenizer_test")
    print(f"\nSaving tokenizer to {save_path}")
    tokenizer.save(save_path)
    
    print(f"Loading tokenizer from {save_path}")
    loaded_tokenizer = BPETokenizer.load(save_path)
    
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
    
    # HTML preprocessing test
    html_text = "<p>This is <b>some HTML</b> that should be cleaned.</p>"
    print(f"\nProcessing HTML text: '{html_text}'")
    
    html_result = tokenizer.encode(html_text, return_tokens=True)
    print("Result after preprocessing:")
    if html_result.tokens:
        print(f"Tokens: {html_result.tokens}")
    decoded_html = tokenizer.decode(html_result.input_ids)
    print(f"Decoded: '{decoded_html}'")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 