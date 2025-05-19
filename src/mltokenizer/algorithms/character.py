# src/mltokenizer/algorithms/character.py
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

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
from mltokenizer.normalization.normalizers import Normalizer
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline

class CharacterTokenizer(BaseTokenizer):
    """A simple character-level tokenizer."""

    def __init__(
        self,
        normalizer: Optional[Normalizer] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
        encoder: Optional[Encoder] = None, # Will be built from character vocab
        special_tokens: Optional[SpecialTokens] = None,
        options: Optional[TokenizerOptions] = None,
        char_vocab_list: Optional[List[str]] = None, # Changed from Set to List for order if provided
        unk_token: str = "<UNK>",
    ):
        st_handler = special_tokens or SpecialTokens(unk_token=unk_token)
        super().__init__(
            tokenizer_type=TokenizerType.CHARACTER,
            vocab_size=0, 
            normalizer=normalizer, 
            preprocessor=preprocessor or PreprocessingPipeline(),
            encoder=encoder,
            special_tokens=st_handler,
            options=options or TokenizerOptions(),
        )
        
        self._is_trained = False
        # No need for _char_to_id, _id_to_char directly on self, managed by self.encoder

        if char_vocab_list:
            self._build_encoder_from_chars(char_vocab_list)
            self._is_trained = True
        elif encoder: 
            self.vocab_size = encoder.vocab_size
            self.special_tokens.sync_with_encoder(encoder)
            self._is_trained = True

        logger.debug(f"Initialized CharacterTokenizer. Trained: {self.is_trained}, Vocab size: {self.vocab_size}")

    def _build_encoder_from_chars(self, unique_chars_list: List[str]):
        """Builds the encoder from a list of unique characters, ensuring special tokens are prioritized."""
        # Start with all defined special tokens to ensure they get the lowest IDs
        # self.special_tokens.all_special_tokens gives only the string values
        final_vocab_list = []
        # Add registered special tokens first, maintaining their order if possible (though Encoder sorts by default if map given)
        # For deterministic IDs, we add them in a specific order.
        # The SpecialTokens handler should provide a canonical list of its tokens for vocab building.
        # For now, using get_special_tokens_dict and filtering None.
        
        # Get all token strings from SpecialTokens handler that are not None
        # This includes default ones like unk, bos, eos, pad, and any custom ones added.
        initial_special_tokens = []
        for token_attr in ["unk_token", "bos_token", "eos_token", "pad_token", "cls_token", "sep_token", "mask_token"]:
            token_val = getattr(self.special_tokens, token_attr, None)
            if token_val:
                initial_special_tokens.append(token_val)
        # Add any other custom special tokens not covered by the fixed attributes
        for token_val in self.special_tokens.get_custom_tokens():
            if token_val not in initial_special_tokens:
                 initial_special_tokens.append(token_val)
        
        final_vocab_list.extend(initial_special_tokens)

        for char_token in unique_chars_list:
            if char_token not in final_vocab_list: # Avoid duplicating special tokens if they were in unique_chars_list
                final_vocab_list.append(char_token)
        
        # Create the token_to_id map for the Encoder
        token_to_id_map = {token: i for i, token in enumerate(final_vocab_list)}
        
        self.encoder = Encoder(token_to_id_map)
        self.vocab_size = self.encoder.vocab_size
        self.special_tokens.sync_with_encoder(self.encoder)
        logger.info(f"Character encoder built. Vocab size: {self.vocab_size}. Special tokens registered: {self.special_tokens.special_token_ids}")

    def train(self, texts: List[str], **kwargs) -> None:
        """Trains the character tokenizer by finding all unique characters in the provided texts."""
        logger.info(f"Training CharacterTokenizer on {len(texts)} texts...")
        unique_chars_set: Set[str] = set()

        for text_item in texts:
            normalized_text = self.normalizer.normalize(text_item) if self.normalizer else text_item
            processed_text = self.preprocessor.process(normalized_text) if self.preprocessor else normalized_text
            unique_chars_set.update(list(processed_text))
        
        self._build_encoder_from_chars(sorted(list(unique_chars_set))) # Sort for deterministic vocab order
        self._is_trained = True
        logger.info(f"CharacterTokenizer training complete. Found {len(unique_chars_set)} unique characters. Total vocab size: {self.vocab_size}")

    def encode(
        self, 
        text: str, 
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        return_tokens: bool = False
    ) -> TokenizedOutput:
        if not self.is_trained or not self.encoder:
            raise UntrainedTokenizerError("Character tokenizer is not trained.")

        normalized_text = self.normalizer.normalize(text) if self.normalizer else text
        processed_text = self.preprocessor.process(normalized_text) if self.preprocessor else normalized_text
        
        chars1 = list(processed_text)
        ids1 = self.encoder.encode(chars1, add_bos_eos=False, add_cls_sep=False) 
        tokens1 = chars1 if return_tokens else None

        ids2: Optional[List[int]] = None
        tokens2: Optional[List[str]] = None
        if text_pair:
            normalized_pair = self.normalizer.normalize(text_pair) if self.normalizer else text_pair
            processed_pair = self.preprocessor.process(normalized_pair) if self.preprocessor else normalized_pair
            chars2 = list(processed_pair)
            ids2 = self.encoder.encode(chars2, add_bos_eos=False, add_cls_sep=False)
            if return_tokens: tokens2 = chars2

        final_ids: List[int] = []
        final_tt_ids: List[int] = [] 
        final_sp_mask: List[int] = [] 
        final_tokens_list: Optional[List[str]] = [] if return_tokens else None

        _sp = self.special_tokens
        is_pair = ids2 is not None

        if add_special_tokens:
            if is_pair and _sp.cls_token_id is not None and _sp.sep_token_id is not None:
                final_ids.append(_sp.cls_token_id); final_tt_ids.append(0); final_sp_mask.append(1);
                if return_tokens and _sp.cls_token: final_tokens_list.append(_sp.cls_token)
                
                final_ids.extend(ids1); final_tt_ids.extend([0]*len(ids1)); final_sp_mask.extend([0]*len(ids1))
                if return_tokens and tokens1: final_tokens_list.extend(tokens1)
                
                final_ids.append(_sp.sep_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                if return_tokens and _sp.sep_token: final_tokens_list.append(_sp.sep_token)

                final_ids.extend(ids2); final_tt_ids.extend([1]*len(ids2)); final_sp_mask.extend([0]*len(ids2))
                if return_tokens and tokens2: final_tokens_list.extend(tokens2)

                final_ids.append(_sp.sep_token_id); final_tt_ids.append(1); final_sp_mask.append(1)
                if return_tokens and _sp.sep_token: final_tokens_list.append(_sp.sep_token)
            else: 
                if _sp.bos_token_id is not None:
                    final_ids.append(_sp.bos_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                    if return_tokens and _sp.bos_token: final_tokens_list.append(_sp.bos_token)
                
                final_ids.extend(ids1); final_tt_ids.extend([0]*len(ids1)); final_sp_mask.extend([0]*len(ids1))
                if return_tokens and tokens1: final_tokens_list.extend(tokens1)

                if is_pair:
                    # Add EOS for first sequence if BOS/EOS mode for pairs
                    if _sp.eos_token_id is not None and not (_sp.cls_token_id and _sp.sep_token_id): 
                         final_ids.append(_sp.eos_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                         if return_tokens and _sp.eos_token: final_tokens_list.append(_sp.eos_token)
                    
                    final_ids.extend(ids2); final_tt_ids.extend([0]*len(ids2)); final_sp_mask.extend([0]*len(ids2)) # Typically TT 0 for char based models or second seq
                    if return_tokens and tokens2: final_tokens_list.extend(tokens2)
                
                if _sp.eos_token_id is not None: 
                    final_ids.append(_sp.eos_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                    if return_tokens and _sp.eos_token: final_tokens_list.append(_sp.eos_token)
        else: 
            final_ids.extend(ids1); final_tt_ids.extend([0]*len(ids1)); final_sp_mask.extend([0]*len(ids1))
            if return_tokens and tokens1: final_tokens_list.extend(tokens1)
            if is_pair:
                final_ids.extend(ids2); final_tt_ids.extend([0]*len(ids2)); final_sp_mask.extend([0]*len(ids2))
                if return_tokens and tokens2: final_tokens_list.extend(tokens2)

        truncated_ids, truncated_tt_ids, overflow = self.apply_truncation(final_ids, token_type_ids=final_tt_ids)
        if return_tokens and final_tokens_list and len(final_tokens_list) > len(truncated_ids): final_tokens_list = final_tokens_list[:len(truncated_ids)]
        if len(final_sp_mask) > len(truncated_ids): final_sp_mask = final_sp_mask[:len(truncated_ids)]

        padded_ids, attention_mask, padded_tt_ids = self.apply_padding(truncated_ids, token_type_ids=truncated_tt_ids)
        if return_tokens and final_tokens_list and len(final_tokens_list) < len(padded_ids):
            # Use the actual pad_token string from SpecialTokens handler
            pad_token_str = _sp.pad_token
            if pad_token_str is None: # Should not happen if pad_id is set, but as a fallback
                logger.warning("Pad token string is None, using default '<PAD_CHAR>'. Ensure pad_token is configured.")
                pad_token_str = "<PAD_CHAR>" # This fallback token might not be in vocab
            final_tokens_list.extend([pad_token_str] * (len(padded_ids) - len(final_tokens_list)))
        if len(final_sp_mask) < len(padded_ids): final_sp_mask.extend([0] * (len(padded_ids) - len(final_sp_mask)))
        
        return TokenizedOutput(
            input_ids=padded_ids,
            attention_mask=attention_mask if self.options.return_attention_mask else None,
            token_type_ids=padded_tt_ids if self.options.return_token_type_ids and any(id_val !=0 for id_val in padded_tt_ids) else None,
            special_tokens_mask=final_sp_mask if self.options.return_special_tokens_mask else None,
            overflowing_tokens=overflow if self.options.return_overflowing_tokens else None,
            length=len(padded_ids) if self.options.return_length else None,
            tokens=final_tokens_list
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self.is_trained or not self.encoder:
            raise UntrainedTokenizerError("Character tokenizer is not trained.")

        ids_to_decode = token_ids
        if skip_special_tokens and self.special_tokens:
            all_sp_ids = self.special_tokens.all_special_ids
            ids_to_decode = [id_val for id_val in token_ids if id_val not in all_sp_ids]
        
        decoded_chars = self.encoder.decode_ids(ids_to_decode) # Changed from self.encoder.decode
        return "".join(decoded_chars) 

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Saves the Character tokenizer (vocabulary and config)."""
        if not self.is_trained or not self.encoder:
            raise TokenizationError("Cannot save Character tokenizer: not trained or no vocabulary.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        ordered_char_list = [self.encoder.id_to_token[i] for i in range(self.encoder.vocab_size)]

        vocab_file = save_dir / "character_vocab.json"
        try:
            with open(vocab_file, "w", encoding="utf-8") as f:
                json.dump(ordered_char_list, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Character vocabulary saved to {vocab_file}")
        except IOError as e:
            self.logger.error(f"Failed to save character vocabulary to {vocab_file}: {e}")
            raise TokenizationError(f"Failed to save character vocabulary: {e}")

        config_data = {
            "tokenizer_type": self.tokenizer_type.value,
            "vocab_size": self.vocab_size,
            "special_tokens_map": self.special_tokens.get_special_tokens_dict(),
            "special_token_ids": self.special_tokens.special_token_ids,
            "options": self.options.dict(),
            "normalizer_config": self.normalizer.config if self.normalizer and hasattr(self.normalizer, 'config') else None,
            "preprocessor_config": self.preprocessor.config if self.preprocessor and hasattr(self.preprocessor, 'config') else None,
        }
        config_file_path = save_dir / "tokenizer_config.json"
        try:
            with open(config_file_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"CharacterTokenizer configuration saved to {config_file_path}")
        except IOError as e:
            self.logger.error(f"Failed to save tokenizer configuration to {config_file_path}: {e}")
            raise TokenizationError(f"Failed to save tokenizer configuration: {e}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "CharacterTokenizer":
        load_dir = Path(path)
        config_file_path = load_dir / "tokenizer_config.json"
        vocab_file = load_dir / "character_vocab.json"

        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file 'tokenizer_config.json' not found in {load_dir}")
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file 'character_vocab.json' not found in {load_dir}")

        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load or parse configuration from {config_file_path}: {e}")
            raise TokenizationError(f"Failed to load tokenizer configuration: {e}")

        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                character_list_from_vocab = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load or parse vocabulary from {vocab_file}: {e}")
            raise TokenizationError(f"Failed to load character vocabulary: {e}")

        # Reconstruct SpecialTokens
        special_tokens_map = config.get("special_tokens_map", {})
        st_handler = SpecialTokens(**special_tokens_map) # IDs will be synced later by _build_encoder_from_chars

        # Reconstruct TokenizerOptions
        options_dict = config.get("options", {})
        tokenizer_options = TokenizerOptions(**options_dict)
        
        # TODO: Reconstruct Normalizer and Preprocessor from their configs
        # normalizer_config = config.get("normalizer_config")
        # preprocessor_config = config.get("preprocessor_config")
        # For now, pass None, or allow override via kwargs
        loaded_normalizer = kwargs.get('normalizer')
        loaded_preprocessor = kwargs.get('preprocessor')

        # Create tokenizer instance. _build_encoder_from_chars will be called, which also syncs special tokens.
        # Pass char_vocab_list for _build_encoder_from_chars to use. This list already has special tokens
        # correctly ordered from the saved vocab file.
        tokenizer = cls(
            special_tokens=st_handler,
            options=tokenizer_options,
            char_vocab_list=character_list_from_vocab, # This list comes from ordered save
            normalizer=loaded_normalizer,
            preprocessor=loaded_preprocessor
            # unk_token is part of special_tokens_map now
        )
        
        # Vocab size is set by _build_encoder_from_chars via __init__
        # _is_trained is set by __init__ because char_vocab_list is provided
        if tokenizer.vocab_size != config.get("vocab_size"):
             logger.warning(f"Loaded vocab size {tokenizer.vocab_size} differs from config {config.get('vocab_size')}. Using actual loaded size.")

        logger.info(f"CharacterTokenizer loaded from {load_dir}. Vocab size: {tokenizer.vocab_size}")
        return tokenizer

__all__ = ["CharacterTokenizer"] 