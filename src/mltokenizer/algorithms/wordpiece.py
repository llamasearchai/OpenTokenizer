# src/mltokenizer/algorithms/wordpiece.py
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from mltokenizer.core.base_tokenizer import (
    BaseTokenizer,
    TokenizedOutput,
    TokenizerOptions,
    TokenizerType,
)
from mltokenizer.core.errors import TokenizationError, UntrainedTokenizerError
from mltokenizer.encoding.encoder import Encoder
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import Normalizer, SequenceNormalizer
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline

class WordpieceTokenizer(BaseTokenizer):
    """WordPiece Tokenizer implementation (Skeleton)."""

    def __init__(
        self,
        vocab_size: int = 30000,
        normalizer: Optional[Normalizer] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
        encoder: Optional[Encoder] = None,
        special_tokens: Optional[SpecialTokens] = None,
        options: Optional[TokenizerOptions] = None,
        vocab: Optional[Dict[str, int]] = None, # WordPiece usually loads a vocab directly
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
    ):
        # Resolve default special tokens
        st_handler = special_tokens or SpecialTokens(
            unk_token=unk_token, 
            # Let SpecialTokens use its defaults for others unless overridden
        )

        super().__init__(
            tokenizer_type=TokenizerType.WORDPIECE,
            vocab_size=vocab_size, 
            normalizer=normalizer or SequenceNormalizer.bert_normalizer(), 
            preprocessor=preprocessor or PreprocessingPipeline(), # Ensure preprocessor is initialized
            encoder=encoder,
            special_tokens=st_handler,
            options=options or TokenizerOptions(),
        )
        self.max_input_chars_per_word = max_input_chars_per_word
        self._is_trained = False 

        if vocab:
            # If a vocab is provided, use it to initialize the encoder
            self.encoder = Encoder(vocab)
            self.vocab_size = self.encoder.vocab_size 
            self._is_trained = True
            # Ensure special tokens from the handler are in the vocab or add them
            # The special_tokens handler should already have the desired tokens (e.g. [UNK])
            # We need to make sure these are in the encoder's vocab and have IDs.
            added_count = self.add_special_tokens(self.special_tokens.get_special_tokens_dict())
            if added_count > 0:
                 logger.info(f"Added {added_count} special tokens to WordPiece vocabulary.")
            # Update vocab size again if special tokens were added and new
            self.vocab_size = self.encoder.vocab_size
        
        self.logger.debug(f"Initialized WordPiece tokenizer. Vocab size: {self.vocab_size}, Trained: {self.is_trained}")

    def train(self, texts: List[str], **kwargs) -> None:
        """WordPiece is typically not trained from scratch in the same way as BPE.
           It often uses a pre-existing vocabulary.
           This method can be implemented if a WordPiece training algorithm is added.
        """
        self.logger.warning(
            "WordPiece training from scratch is not standard. "
            "Usually loads a pre-built vocab or uses a generation script (e.g., from BERT prep). "
            "This 'train' method is a placeholder for basic vocab building if no vocab is loaded."
        )
        if self.encoder and self._is_trained:
            logger.info("Tokenizer already has a vocabulary/encoder. Skipping training.")
            return

        # Placeholder: basic vocab building for demonstration.
        # This is NOT how actual WordPiece subword vocabulary is built.
        from collections import Counter
        
        # 1. Pre-tokenize text (simple whitespace split for this placeholder)
        all_words_flat = []
        for text_item in texts:
            normalized_text = self.normalizer.normalize(text_item) if self.normalizer else text_item
            preprocessed_text = self.preprocessor.process(normalized_text) if self.preprocessor else normalized_text
            all_words_flat.extend(preprocessed_text.split())

        # 2. Count word frequencies
        word_counts = Counter(all_words_flat)
        
        # 3. Create initial character vocabulary from all words
        char_vocab = set()
        for word in word_counts:
            char_vocab.update(list(word))
        
        # 4. Iteratively build subword vocabulary (highly simplified placeholder)
        # Actual WordPiece uses a likelihood-based approach on a corpus.
        # This example just takes most frequent words as "subwords" for simplicity.
        
        # Ensure special tokens are part of the initial vocab list
        current_vocab_list = list(self.special_tokens.all_special_tokens)
        
        # Add most frequent words to fill up vocab_size (minus special tokens)
        num_to_add = self.vocab_size - len(current_vocab_list)
        if num_to_add > 0:
            for word, _ in word_counts.most_common(num_to_add):
                if word not in current_vocab_list: # Avoid duplicates
                    current_vocab_list.append(word)
                    if len(current_vocab_list) >= self.vocab_size:
                        break
        
        # Create token_to_id mapping
        token_to_id_map = {token: i for i, token in enumerate(current_vocab_list)}
        
        self.encoder = Encoder(token_to_id_map)
        self.vocab_size = self.encoder.vocab_size # Update with actual size
        self.add_special_tokens(self.special_tokens.get_special_tokens_dict()) # Re-ensure and get IDs
        self.vocab_size = self.encoder.vocab_size # Final update
        
        self._is_trained = True
        self.logger.info(f"WordPiece tokenizer 'trained' (placeholder vocab built). Vocab size: {self.vocab_size}")


    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenizes a single word into WordPieces."""
        if not self.is_trained or not self.encoder:
            raise UntrainedTokenizerError("WordPiece tokenizer is not trained or vocab not loaded.")

        if word in self.encoder.token_to_id_map:
            return [word]

        if len(word) > self.max_input_chars_per_word:
            return [self.special_tokens.unk_token]

        tokens = []
        current_pos = 0
        word_len = len(word)

        while current_pos < word_len:
            # Find the longest matching subword from the end of the remaining part
            best_match_len = 0
            best_subword = None
            
            # Iterate from longest possible subword to shortest
            for end_pos in range(word_len, current_pos, -1):
                subword_candidate = word[current_pos:end_pos]
                # WordPiece uses "##" prefix for subwords not at the beginning of the original word
                if current_pos > 0:
                    subword_to_check = "##" + subword_candidate
                else:
                    subword_to_check = subword_candidate
                
                if subword_to_check in self.encoder.token_to_id_map:
                    best_subword = subword_to_check
                    best_match_len = len(subword_candidate) # length of the original part, not the "##" prefixed one
                    break
            
            if best_subword and best_match_len > 0:
                tokens.append(best_subword)
                current_pos += best_match_len
            else:
                # If no subword found (even a single character), it's an UNK situation for this part.
                # This part of the logic implies that all individual characters (possibly ##prefixed)
                # should be in the vocabulary if we want to avoid UNKing parts of words.
                # For simplicity, if we can't tokenize the rest, we consider the whole original word UNK.
                # A more robust implementation might add characters to vocab or handle UNK more granularly.
                return [self.special_tokens.unk_token] 
        
        return tokens if tokens else [self.special_tokens.unk_token]


    def encode(
        self, 
        text: str, 
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True, # Consistent with BaseTokenizer
        return_tokens: bool = False  # Consistent with BaseTokenizer
    ) -> TokenizedOutput:
        if not self.is_trained or not self.encoder:
            raise UntrainedTokenizerError("WordPiece tokenizer is not trained/vocab not loaded.")

        # 1. Normalization
        normalized_text = self.normalizer.normalize(text) if self.normalizer else text
        normalized_text_pair = None
        if text_pair:
            normalized_text_pair = self.normalizer.normalize(text_pair) if self.normalizer else text_pair

        # 2. Preprocessing (if any, beyond normalization)
        processed_text = self.preprocessor.process(normalized_text) if self.preprocessor else normalized_text
        processed_text_pair = None
        if normalized_text_pair:
            processed_text_pair = self.preprocessor.process(normalized_text_pair) if self.preprocessor else normalized_text_pair
            
        # 3. Pre-tokenization (splitting text into "words" before WordPiece algorithm)
        # BERT's BasicTokenizer typically handles whitespace splitting and punctuation.
        # This is a simplified version:
        raw_words = processed_text.split() # Simple whitespace split
        
        final_tokens = []
        for word in raw_words:
            final_tokens.extend(self._tokenize_word(word))
        
        text_ids = self.encoder.encode(final_tokens)
        output_tokens = final_tokens if return_tokens else None

        text_pair_ids = []
        pair_output_tokens = []
        if processed_text_pair:
            raw_words_pair = processed_text_pair.split()
            for word in raw_words_pair:
                pair_output_tokens.extend(self._tokenize_word(word))
            text_pair_ids = self.encoder.encode(pair_output_tokens)
            if return_tokens and output_tokens is not None: # output_tokens must exist if pair_output_tokens are to be appended
                 output_tokens.extend(pair_output_tokens)
            elif return_tokens and output_tokens is None: # if text was empty but text_pair is not
                 output_tokens = pair_output_tokens


        # --- Combine and add special tokens, truncate, pad ---
        # This logic is largely copied from BaseTokenizer's encode_batch and adapted
        # It can be refactored into BaseTokenizer for better DRY principle.

        input_ids_sequence_1 = text_ids
        input_ids_sequence_2 = text_pair_ids if text_pair is not None else None
        
        # Initial token type IDs (before special tokens)
        token_type_ids = [0] * len(input_ids_sequence_1)
        if input_ids_sequence_2 is not None:
            token_type_ids.extend([1] * len(input_ids_sequence_2))

        # Combine sequences if pair exists
        combined_input_ids = input_ids_sequence_1
        if input_ids_sequence_2 is not None:
            combined_input_ids.extend(input_ids_sequence_2)

        # Add special tokens
        final_special_tokens_mask = [0] * len(combined_input_ids) # Initialize before modification
        if add_special_tokens and self.special_tokens:
            # This part assumes BERT-like special tokens ([CLS] seq [SEP] seq_pair [SEP])
            # It should be made more generic in BaseTokenizer or configurable.
            cls_token_id = self.special_tokens.cls_token_id
            sep_token_id = self.special_tokens.sep_token_id

            _input_ids = []
            _token_type_ids = []
            _special_tokens_mask = []
            _output_tokens_list = [] # For string tokens

            # Optional CLS token at the beginning
            if cls_token_id is not None:
                _input_ids.append(cls_token_id)
                _token_type_ids.append(0) # CLS is type 0
                _special_tokens_mask.append(1)
                if return_tokens: _output_tokens_list.append(self.special_tokens.cls_token)

            # First sequence
            _input_ids.extend(input_ids_sequence_1)
            _token_type_ids.extend([0] * len(input_ids_sequence_1))
            _special_tokens_mask.extend([0] * len(input_ids_sequence_1))
            if return_tokens: _output_tokens_list.extend(final_tokens)


            # Separator and second sequence (if pair exists)
            if input_ids_sequence_2 is not None:
                if sep_token_id is not None:
                    _input_ids.append(sep_token_id)
                    _token_type_ids.append(0) # SEP after first seq is type 0
                    _special_tokens_mask.append(1)
                    if return_tokens: _output_tokens_list.append(self.special_tokens.sep_token)

                _input_ids.extend(input_ids_sequence_2)
                _token_type_ids.extend([1] * len(input_ids_sequence_2)) # Second sequence is type 1
                _special_tokens_mask.extend([0] * len(input_ids_sequence_2))
                if return_tokens: _output_tokens_list.extend(pair_output_tokens)
            
            # Final SEP token (always, if SEP is defined)
            if sep_token_id is not None:
                # Determine type of final SEP: 0 if no pair, 1 if pair
                final_sep_type = 1 if input_ids_sequence_2 is not None else 0
                _input_ids.append(sep_token_id)
                _token_type_ids.append(final_sep_type) 
                _special_tokens_mask.append(1)
                if return_tokens: _output_tokens_list.append(self.special_tokens.sep_token)

            combined_input_ids = _input_ids
            token_type_ids = _token_type_ids
            final_special_tokens_mask = _special_tokens_mask
            if return_tokens: output_tokens = _output_tokens_list
        
        # Truncation
        truncated_input_ids, truncated_token_type_ids, overflowing_tokens = self.apply_truncation(
            combined_input_ids, 
            token_type_ids=token_type_ids
            # No pair_input_ids here as they are already combined
        )
        # Adjust string tokens and special_tokens_mask if truncation happened
        if return_tokens and output_tokens and len(output_tokens) > len(truncated_input_ids):
            output_tokens = output_tokens[:len(truncated_input_ids)] # Simple truncation
        if len(final_special_tokens_mask) > len(truncated_input_ids):
            final_special_tokens_mask = final_special_tokens_mask[:len(truncated_input_ids)]


        # Padding
        padded_input_ids, attention_mask, padded_token_type_ids = self.apply_padding(
            truncated_input_ids, 
            token_type_ids=truncated_token_type_ids
        )
        # Adjust string tokens and special_tokens_mask if padding happened
        if return_tokens and output_tokens and len(output_tokens) < len(padded_input_ids):
             pad_token_str = self.special_tokens.pad_token
             output_tokens.extend([pad_token_str] * (len(padded_input_ids) - len(output_tokens)))
        if len(final_special_tokens_mask) < len(padded_input_ids):
            # Padding implies these are not special tokens, mask should be 0
            final_special_tokens_mask.extend([0] * (len(padded_input_ids) - len(final_special_tokens_mask)))


        return TokenizedOutput(
            input_ids=padded_input_ids,
            attention_mask=attention_mask if self.options.return_attention_mask else None,
            token_type_ids=padded_token_type_ids if self.options.return_token_type_ids else None,
            special_tokens_mask=final_special_tokens_mask if self.options.return_special_tokens_mask else None,
            overflowing_tokens=overflowing_tokens if self.options.return_overflowing_tokens else None,
            length=len(padded_input_ids) if self.options.return_length else None,
            tokens=output_tokens if return_tokens else None # final adjusted list of string tokens
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self.is_trained or not self.encoder:
            raise UntrainedTokenizerError("WordPiece tokenizer is not trained/vocab not loaded.")

        # Filter special tokens if requested
        actual_ids_to_decode = token_ids
        if skip_special_tokens and self.special_tokens:
            actual_ids_to_decode = [id_val for id_val in token_ids if id_val not in self.special_tokens.all_special_ids]
        
        tokens = self.encoder.decode(actual_ids_to_decode)
        
        # Basic detokenization for WordPiece
        # Concatenate tokens, removing '##' prefixes and handling spaces.
        # This is a common way for BERT-style WordPiece.
        output_string = ""
        for token in tokens:
            if token.startswith("##"):
                output_string += token[2:]
            else:
                if output_string: # Add space if not the first token and previous didn't end with space
                    output_string += " "
                output_string += token
        
        # Further post-processing might be needed (e.g., true casing, punctuation spacing)
        # For example, a more sophisticated detokenizer might use the original text context or rules.
        return output_string.strip()


    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Saves the WordPiece tokenizer (vocabulary and config)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if not self.encoder:
            raise TokenizationError("Cannot save WordPiece tokenizer without an encoder/vocabulary.")

        # Save vocabulary (standard for WordPiece is a vocab.txt file, one token per line)
        vocab_file = path / "vocab.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            # Sort by ID for consistent vocab.txt
            for token, _ in sorted(self.encoder.get_vocab().items(), key=lambda item: item[1]):
                f.write(token + "\n")
        self.logger.info(f"Vocabulary saved to {vocab_file}")
        
        # Save tokenizer configuration
        config = {
            "tokenizer_type": self.tokenizer_type.value,
            "vocab_size": self.vocab_size, # This is the vocab_size from init, might differ from actual if loaded
            "max_input_chars_per_word": self.max_input_chars_per_word,
            # Serialize special_tokens by getting their string values
            "special_tokens_map": self.special_tokens.get_special_tokens_dict() if self.special_tokens else {},
            "options": self.options.dict() if self.options else {},
            # Normalizer and Preprocessor state would need custom serialization if they have configurable state
            "normalizer_config": self.normalizer.normalize.__class__.__name__ if self.normalizer else None, # Example
        }
        config_file = path / "tokenizer_config.json" # More specific name
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Tokenizer configuration saved to {config_file}")
        
        self.logger.info(f"Saved WordPiece tokenizer to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "WordpieceTokenizer":
        """Loads a WordPiece tokenizer from a directory."""
        path = Path(path)
        
        # 1. Load configuration
        config_file = path / "tokenizer_config.json"
        if not config_file.exists():
            # Fallback for older general "config.json" name
            config_file = path / "config.json" 
            if not config_file.exists():
                 raise FileNotFoundError(f"Tokenizer configuration file (tokenizer_config.json or config.json) not found in {path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 2. Load vocabulary (standard vocab.txt for WordPiece/BERT)
        vocab_file = path / "vocab.txt"
        if not vocab_file.exists():
            raise FileNotFoundError(f"vocab.txt not found in {path}")
        
        loaded_vocab: Dict[str, int] = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token: # Ensure non-empty lines
                    loaded_vocab[token] = i # ID is often implicit by line number in vocab.txt
        
        # 3. Reconstruct SpecialTokens handler
        special_tokens_map = config.get("special_tokens_map", {})
        # Convert map to kwargs for SpecialTokens constructor where possible
        st_kwargs = {key: val for key, val in special_tokens_map.items() if hasattr(SpecialTokens, key)}
        # Any additional special tokens not part of constructor args
        additional_st = {key: val for key, val in special_tokens_map.items() if not hasattr(SpecialTokens, key)}
        
        special_tokens_handler = SpecialTokens(**st_kwargs)
        if additional_st: # Register any others
            special_tokens_handler.register_special_tokens(additional_st)

        # 4. Reconstruct TokenizerOptions
        options_obj = TokenizerOptions(**config.get("options", {}))
        
        # 5. Reconstruct Normalizer and Preprocessor (if config exists)
        # This part is simplified; real normalizers/preprocessors might need more complex deserialization.
        normalizer_name = config.get("normalizer_config")
        normalizer_instance = None
        if normalizer_name == "BertNormalizer": # Example based on SequenceNormalizer factory
            normalizer_instance = SequenceNormalizer.bert_normalizer()
        elif normalizer_name == "GPTNormalizer":
            normalizer_instance = SequenceNormalizer.gpt_normalizer()
        elif normalizer_name: # Could try to dynamically load based on class name
            logger.warning(f"Cannot automatically reconstruct normalizer: {normalizer_name}. Using default.")
            normalizer_instance = SequenceNormalizer.bert_normalizer() # Fallback
        else: # Default if not specified
             normalizer_instance = SequenceNormalizer.bert_normalizer()


        # 6. Create the tokenizer instance
        # The __init__ of WordpieceTokenizer will handle creating the Encoder from 'loaded_vocab'
        tokenizer = cls(
            vocab_size=config.get("vocab_size", len(loaded_vocab)), # Best guess for original target vocab_size
            vocab=loaded_vocab, # Pass the loaded vocab to __init__
            normalizer=normalizer_instance,
            # preprocessor= ... # Load preprocessor similarly if saved
            special_tokens=special_tokens_handler,
            options=options_obj,
            max_input_chars_per_word=config.get("max_input_chars_per_word", 100),
        )
        # The __init__ method sets _is_trained = True if vocab is provided.
        # It also re-adds special tokens to the encoder and updates IDs in special_tokens_handler.
        
        logger.info(f"Loaded WordPiece tokenizer from {path}. Actual vocab size: {tokenizer.encoder.vocab_size}")
        return tokenizer

__all__ = ["WordpieceTokenizer"] 