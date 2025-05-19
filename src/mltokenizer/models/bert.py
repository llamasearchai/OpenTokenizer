from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
from mltokenizer.core.base_tokenizer import TokenizerOptions
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import SequenceNormalizer, LowercaseNormalizer


class BertTokenizer:
    """Adapter for BERT tokenizer."""
    
    @staticmethod
    def from_pretrained(
        model_name_or_path: Union[str, Path],
        do_lower_case: bool = True,
        max_length: Optional[int] = None,
        **kwargs
    ) -> WordpieceTokenizer:
        """Load a BERT tokenizer from a pretrained model.
        
        Args:
            model_name_or_path: Model name or path
            do_lower_case: Whether to lowercase text
            max_length: Maximum sequence length
            **kwargs: Additional parameters
            
        Returns:
            WordpieceTokenizer instance
        """
        logger.info(f"Loading BERT tokenizer from {model_name_or_path}")
        
        # Determine if it's a local path or a model name
        path = Path(model_name_or_path)
        if path.exists() and path.is_dir():
            # Load from local path
            try:
                tokenizer = WordpieceTokenizer.load(path)
                logger.info(f"Loaded BERT tokenizer from {path}")
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load BERT tokenizer from {path}: {e}")
                logger.info("Falling back to creating a new tokenizer")
        
        # Create a new tokenizer with BERT defaults
        special_tokens = SpecialTokens(
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )
        
        options = TokenizerOptions(
            add_bos_token=False,
            add_eos_token=False,
            add_padding_token=True,
            truncation_strategy="longest_first",
            max_length=max_length or 512,
            padding_strategy="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_special_tokens_mask=False
        )
        
        # Use BERT-specific normalizer
        normalizer = SequenceNormalizer.bert_normalizer()
        if do_lower_case:
            # Add lowercase to the normalizer sequence
            normalizer.normalizers.insert(1, LowercaseNormalizer())
        
        # Initialize the tokenizer
        tokenizer = WordpieceTokenizer(
            vocab_size=30522,  # Default BERT vocab size
            normalizer=normalizer,
            special_tokens=special_tokens,
            options=options
        )
        
        logger.info(f"Created new BERT tokenizer with vocab size {tokenizer.vocab_size}")
        
        # Try to load vocab and merges if available
        if kwargs.get("vocab_file"):
            vocab_file = kwargs["vocab_file"]
            logger.info(f"Loading vocabulary from {vocab_file}")
            # Implementation for loading vocabulary from file
        
        return tokenizer
    
    @staticmethod
    def from_huggingface(
        model_name: str,
        local_files_only: bool = False,
        **kwargs
    ) -> WordpieceTokenizer:
        """Load a BERT tokenizer from Hugging Face.
        
        Args:
            model_name: Model name on Hugging Face
            local_files_only: Whether to only use local files
            **kwargs: Additional parameters
            
        Returns:
            WordpieceTokenizer instance
        """
        try:
            from huggingface_hub import snapshot_download
            from transformers import BertTokenizer as HFBertTokenizer
            
            # Download the model
            if not local_files_only:
                logger.info(f"Downloading BERT tokenizer {model_name} from Hugging Face")
                path = snapshot_download(model_name, local_files_only=local_files_only)
            else:
                path = model_name
            
            # Load the HF tokenizer to extract configuration
            hf_tokenizer = HFBertTokenizer.from_pretrained(path)
            
            # Extract vocabulary
            vocab = hf_tokenizer.get_vocab()
            
            # Create options
            special_tokens = SpecialTokens(
                pad_token=hf_tokenizer.pad_token,
                unk_token=hf_tokenizer.unk_token,
                cls_token=hf_tokenizer.cls_token,
                sep_token=hf_tokenizer.sep_token,
                mask_token=hf_tokenizer.mask_token
            )
            
            options = TokenizerOptions(
                add_bos_token=False,
                add_eos_token=False,
                add_padding_token=True,
                truncation_strategy="longest_first",
                max_length=hf_tokenizer.model_max_length,
                padding_strategy="max_length",
                return_attention_mask=True,
                return_token_type_ids=True,
                return_special_tokens_mask=False
            )
            
            # Create normalizer based on whether the tokenizer is cased or not
            do_lower_case = not hf_tokenizer.do_lower_case
            normalizer = SequenceNormalizer.bert_normalizer()
            if do_lower_case:
                normalizer.normalizers.insert(1, LowercaseNormalizer())
            
            # Initialize and train the tokenizer
            tokenizer = WordpieceTokenizer(
                vocab_size=len(vocab),
                normalizer=normalizer,
                special_tokens=special_tokens,
                options=options
            )
            
            # Set the vocabulary directly
            tokenizer.encoder.token_to_id_map = vocab
            tokenizer.encoder.id_to_token_map = {v: k for k, v in vocab.items()}
            tokenizer._is_trained = True
            
            # Register special token IDs
            for token_type in ["pad_token", "unk_token", "cls_token", "sep_token", "mask_token"]:
                token = getattr(hf_tokenizer, token_type)
                token_id = hf_tokenizer.convert_tokens_to_ids([token])[0]
                special_tokens.register_special_token_id(token_type, token_id)
            
            logger.info(f"Successfully loaded BERT tokenizer {model_name} from Hugging Face")
            return tokenizer
            
        except ImportError:
            logger.error("transformers and huggingface_hub packages are required to load from Hugging Face")
            raise
        except Exception as e:
            logger.error(f"Failed to load BERT tokenizer from Hugging Face: {e}")
            raise 