#!/usr/bin/env python
"""
Model Adapters Demo

This script demonstrates how to use the model adapters to load pretrained
tokenizers from HuggingFace models. It shows how to load BERT and GPT tokenizers
and compare their tokenization outputs.
"""

from mltokenizer.models.bert import BertTokenizer
from mltokenizer.models.gpt import GPTTokenizer


def compare_tokens(tokens1, tokens2, label1="Model 1", label2="Model 2"):
    """Compare and display differences between two tokenizations."""
    print(f"\n{label1} tokens ({len(tokens1)}):")
    print(f"  {tokens1}")
    
    print(f"\n{label2} tokens ({len(tokens2)}):")
    print(f"  {tokens2}")
    
    # Calculate overlap
    common = set(tokens1).intersection(set(tokens2))
    only_in_1 = set(tokens1).difference(set(tokens2))
    only_in_2 = set(tokens2).difference(set(tokens1))
    
    overlap_percent = len(common) / len(set(tokens1).union(set(tokens2))) * 100
    
    print("\nTokenization comparison:")
    print(f"  Common tokens: {len(common)} ({overlap_percent:.1f}% overlap)")
    
    if only_in_1:
        print(f"  Tokens only in {label1}: {only_in_1}")
    
    if only_in_2:
        print(f"  Tokens only in {label2}: {only_in_2}")


def main():
    """Run the model adapters demo."""
    print("Model Adapters Demo")
    print("==================\n")
    
    # Note: This downloads models from HuggingFace if they're not cached
    print("Loading pretrained tokenizers (downloading models if needed)...")
    
    # Load BERT tokenizer
    try:
        print("\nLoading BERT tokenizer...")
        bert_tokenizer = BertTokenizer.from_huggingface("bert-base-uncased")
        print(f"Loaded BERT tokenizer with vocabulary size: {bert_tokenizer.vocab_size}")
        bert_loaded = True
    except Exception as e:
        print(f"Error loading BERT tokenizer: {e}")
        bert_loaded = False
    
    # Load GPT tokenizer
    try:
        print("\nLoading GPT-2 tokenizer...")
        gpt_tokenizer = GPTTokenizer.from_huggingface("gpt2")
        print(f"Loaded GPT-2 tokenizer with vocabulary size: {gpt_tokenizer.vocab_size}")
        gpt_loaded = True
    except Exception as e:
        print(f"Error loading GPT-2 tokenizer: {e}")
        gpt_loaded = False
    
    if not bert_loaded and not gpt_loaded:
        print("\nNo tokenizers could be loaded. Please check your internet connection and HuggingFace access.")
        return
    
    # Compare tokenization if both tokenizers are loaded
    if bert_loaded and gpt_loaded:
        print("\nComparing tokenization between BERT and GPT-2:")
        
        # Test texts to compare
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models process text as numerical data.",
            "Tokenization is the first step in preparing text for language models.",
            "Different models use different tokenization strategies, which affects how they handle text.",
            "Artificial intelligence is transforming how we interact with technology.",
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n\n--- Test Text {i+1} ---")
            print(f"Text: '{text}'")
            
            # Tokenize with each model
            bert_result = bert_tokenizer.encode(text, return_tokens=True)
            gpt_result = gpt_tokenizer.encode(text, return_tokens=True)
            
            # Compare tokens
            compare_tokens(
                bert_result.tokens or [], 
                gpt_result.tokens or [],
                "BERT",
                "GPT-2"
            )
    
    # Demonstrate BERT tokenizer if loaded
    if bert_loaded:
        print("\n\n--- BERT Tokenizer Special Features ---")
        
        # BERT sequence pair handling
        text_a = "This is the first sentence."
        text_b = "This is the second paired sentence."
        
        print(f"\nBERT sequence pair encoding: '{text_a}' + '{text_b}'")
        bert_pair_result = bert_tokenizer.encode(text_a, text_pair=text_b, add_special_tokens=True, return_tokens=True)
        
        if bert_pair_result.tokens and bert_pair_result.token_type_ids:
            print("\nBERT sequence pair tokenization:")
            print("Token\t\tType ID")
            print("-----\t\t-------")
            for token, type_id in zip(bert_pair_result.tokens, bert_pair_result.token_type_ids):
                # Adjust spacing for better formatting
                space = "\t\t" if len(token) < 8 else "\t"
                print(f"{token}{space}{type_id}")
        
        # Show special tokens
        print("\nBERT special tokens:")
        for token_type, token in bert_tokenizer.special_tokens.get_special_tokens_dict().items():
            if token:
                print(f"  {token_type}: {token}")
    
    # Demonstrate GPT tokenizer if loaded
    if gpt_loaded:
        print("\n\n--- GPT-2 Tokenizer Special Features ---")
        
        # GPT BPE tokenization examples
        test_text = "The transformer architecture revolutionized NLP."
        print(f"\nGPT-2 tokenization example: '{test_text}'")
        
        gpt_result = gpt_tokenizer.encode(test_text, return_tokens=True)
        
        if gpt_result.tokens:
            print("\nTokenization with token IDs:")
            for token, token_id in zip(gpt_result.tokens, gpt_result.input_ids):
                print(f"  '{token}': {token_id}")
        
        # Show how GPT handles special characters
        special_text = "GPT-2 handles emojis 😊 and special characters: &^%$#@!"
        print(f"\nGPT-2 handling of special characters: '{special_text}'")
        
        special_result = gpt_tokenizer.encode(special_text, return_tokens=True)
        
        if special_result.tokens:
            print("\nSpecial character tokenization:")
            for token in special_result.tokens:
                print(f"  '{token}'")
        
        # Show GPT special tokens
        print("\nGPT-2 special tokens:")
        for token_type, token in gpt_tokenizer.special_tokens.get_special_tokens_dict().items():
            if token:
                print(f"  {token_type}: {token}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 