from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from ..core.base_tokenizer import BaseTokenizer, TokenizerOptions
from ..core.registry import TokenizerRegistry # To potentially load underlying algorithm
from ..encoding.special_tokens import SpecialTokens
from ..normalization.normalizers import Normalizer, SequenceNormalizer


class CustomTokenizerLoader:
    """Adapter for loading custom tokenizers from configuration."""

    @staticmethod
    def from_config(
        config_path_or_dict: Union[str, Path, Dict[str, Any]],
        **kwargs
    ) -> BaseTokenizer:
        """Load a custom tokenizer from a configuration file or dictionary.

        The configuration should specify the tokenizer algorithm (e.g., BPE, WordPiece),
        paths to vocab/merges files, special tokens, normalization pipeline, etc.

        Args:
            config_path_or_dict: Path to a JSON/YAML config file or a config dictionary.
            **kwargs: Additional override parameters.

        Returns:
            A configured tokenizer instance.
        """
        logger.info(f"Loading custom tokenizer from configuration: {config_path_or_dict}")

        config: Dict[str, Any] = {}
        if isinstance(config_path_or_dict, (str, Path)):
            # Placeholder: Load config from YAML or JSON file
            # import yaml or json
            # with open(config_path_or_dict, 'r') as f:
            #     config = yaml.safe_load(f) # or json.load(f)
            logger.warning(f"Loading from file path {config_path_or_dict} is not yet implemented for CustomTokenizerLoader.")
            # For now, assume it might be a model ID that needs other loaders
            pass # Fall through to allow kwargs to define it for now
        elif isinstance(config_path_or_dict, dict):
            config = config_path_or_dict
        else:
            raise ValueError("config_path_or_dict must be a path or a dictionary.")

        # Override config with kwargs
        config.update(kwargs)

        tokenizer_type = config.get("tokenizer_type")
        if not tokenizer_type:
            raise ValueError("Custom tokenizer configuration must specify 'tokenizer_type'.")

        vocab_file = config.get("vocab_file")
        merges_file = config.get("merges_file") # For BPE
        model_file = config.get("model_file") # For SentencePiece/Unigram

        # Placeholder for SpecialTokens configuration
        special_tokens_config = config.get("special_tokens", {})
        special_tokens = SpecialTokens(**special_tokens_config)

        # Placeholder for TokenizerOptions configuration
        options_config = config.get("options", {})
        options = TokenizerOptions(**options_config)
        
        # Placeholder for Normalizer configuration
        # normalizer_config = config.get("normalizer_pipeline", [])
        # normalizers_list = [] # build this list based on config
        # normalizer = ComposeNormalizer(normalizers_list) if normalizers_list else SequenceNormalizer.default()
        normalizer = SequenceNormalizer.default() # Fallback
        
        vocab_size = config.get("vocab_size", 30000) # Default, should come from vocab ideally

        # Instantiate the underlying tokenizer algorithm
        # This part needs to be robust and select the correct class from mltokenizer.algorithms
        registry = TokenizerRegistry() # Temporary instance for this example

        # This is a very simplified instantiation logic. A real one would be more robust,
        # possibly using registry.get_tokenizer_class(tokenizer_type) and then .load() or direct init.
        tokenizer: Optional[BaseTokenizer] = None
        
        if tokenizer_type.lower() == "bpe":
            from ..algorithms.bpe import BPETokenizer
            # BPETokenizer.load() would be ideal if vocab/merges are structured as a saved tokenizer
            # Otherwise, need to load vocab/merges and initialize manually.
            logger.info("Attempting to initialize custom BPETokenizer.")
            # This requires vocab and merges to be loaded and passed if not using .load()
            # For a truly custom load, BPETokenizer would need a method like:
            # .from_files(vocab_file=..., merges_file=..., special_tokens=..., options=..., normalizer=...)
            tokenizer = BPETokenizer(vocab_size=vocab_size, special_tokens=special_tokens, options=options, normalizer=normalizer)
            if vocab_file: # and merges_file for BPE
                 logger.warning("Custom BPETokenizer loading from individual vocab/merges files needs full implementation.")
            tokenizer._is_trained = True # Assume configured means trained for custom

        elif tokenizer_type.lower() == "wordpiece":
            from ..algorithms.wordpiece import WordpieceTokenizer
            logger.info("Attempting to initialize custom WordpieceTokenizer.")
            tokenizer = WordpieceTokenizer(vocab_size=vocab_size, special_tokens=special_tokens, options=options, normalizer=normalizer)
            if vocab_file:
                logger.warning("Custom WordpieceTokenizer loading from vocab_file needs full implementation.")
            tokenizer._is_trained = True

        elif tokenizer_type.lower() == "sentencepiece":
            from ..algorithms.sentencepiece_tokenizer import SentencePieceTokenizer
            if not model_file:
                raise ValueError("SentencePiece custom tokenizer requires 'model_file' in config.")
            logger.info(f"Attempting to initialize custom SentencePieceTokenizer from {model_file}.")
            tokenizer = SentencePieceTokenizer(
                model_path=Path(model_file), 
                vocab_size=vocab_size, # Often derived from model, but can be set
                special_tokens=special_tokens, 
                options=options, 
                normalizer=normalizer
            )
            tokenizer._is_trained = True # SentencePiece model implies trained state
        
        # Add other types like Unigram, Character etc.

        if tokenizer is None:
            raise ValueError(f"Unsupported custom tokenizer_type: {tokenizer_type}")

        logger.info(f"Successfully loaded custom {tokenizer_type} tokenizer.")
        return tokenizer 