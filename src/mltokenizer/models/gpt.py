from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.core.base_tokenizer import TokenizerOptions
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import SequenceNormalizer


class GPTTokenizer:
    """Adapter for GPT tokenizer."""
    
    @staticmethod
    def from_pretrained(
        model_name_or_path: Union[str, Path],
        max_length: Optional[int] = None,
        **kwargs
    ) -> BPETokenizer:
        """Load a GPT tokenizer from a pretrained model.
        
        Args:
            model_name_or_path: Model name or path
            max_length: Maximum sequence length
            **kwargs: Additional parameters
            
        Returns:
            BPETokenizer instance
        """
        logger.info(f"Loading GPT tokenizer from {model_name_or_path}")
        
        # Determine if it's a local path or a model name
        path = Path(model_name_or_path)
        if path.exists() and path.is_dir():
            # Load from local path
            try:
                tokenizer = BPETokenizer.load(path)
                logger.info(f"Loaded GPT tokenizer from {path}")
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load GPT tokenizer from {path}: {e}")
                logger.info("Falling back to creating a new tokenizer")
        
        # Create a new tokenizer with GPT defaults
        special_tokens = SpecialTokens(
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>"
        )
        
        options = TokenizerOptions(
            add_bos_token=True,
            add_eos_token=True,
            add_padding_token=True,
            truncation_strategy="longest_first",
            max_length=max_length or 1024,
            padding_strategy="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_special_tokens_mask=False
        )
        
        # Use GPT-specific normalizer
        normalizer = SequenceNormalizer.gpt_normalizer()
        
        # Initialize the tokenizer
        tokenizer = BPETokenizer(
            vocab_size=50257,  # Default GPT-2 vocab size
            normalizer=normalizer,
            special_tokens=special_tokens,
            options=options
        )
        
        logger.info(f"Created new GPT tokenizer with vocab size {tokenizer.vocab_size}")
        
        return tokenizer
    
    @staticmethod
    def from_huggingface(
        model_name: str,
        local_files_only: bool = False,
        **kwargs
    ) -> BPETokenizer:
        """Load a GPT tokenizer from Hugging Face.
        
        Args:
            model_name: Model name on Hugging Face
            local_files_only: Whether to only use local files
            **kwargs: Additional parameters
            
        Returns:
            BPETokenizer instance
        """
        try:
            from huggingface_hub import snapshot_download
            from transformers import GPT2Tokenizer, GPT2TokenizerFast
            
            # Download the model
            if not local_files_only:
                logger.info(f"Downloading GPT tokenizer {model_name} from Hugging Face")
                path = snapshot_download(model_name, local_files_only=local_files_only)
            else:
                path = model_name
            
            # Try fast tokenizer first, fall back to slower one
            try:
                hf_tokenizer = GPT2TokenizerFast.from_pretrained(path)
            except:
                hf_tokenizer = GPT2Tokenizer.from_pretrained(path)
            
            # Extract vocabulary and merges
            vocab = hf_tokenizer.get_vocab()
            
            # Create options
            special_tokens = SpecialTokens(
                pad_token=hf_tokenizer.pad_token or "<pad>",
                unk_token=hf_tokenizer.unk_token or "<unk>",
                bos_token=hf_tokenizer.bos_token or "<s>",
                eos_token=hf_tokenizer.eos_token or "</s>"
            )
            
            options = TokenizerOptions(
                add_bos_token=True,
                add_eos_token=True,
                add_padding_token=True,
                truncation_strategy="longest_first",
                max_length=hf_tokenizer.model_max_length,
                padding_strategy="max_length",
                return_attention_mask=True,
                return_token_type_ids=False,
                return_special_tokens_mask=False
            )
            
            # Use GPT-specific normalizer
            normalizer = SequenceNormalizer.gpt_normalizer()
            
            # Initialize the tokenizer
            tokenizer = BPETokenizer(
                vocab_size=len(vocab),
                normalizer=normalizer,
                special_tokens=special_tokens,
                options=options
            )
            
            # Set the vocabulary directly
            tokenizer.encoder.token_to_id_map = vocab
            tokenizer.encoder.id_to_token_map = {v: k for k, v in vocab.items()}
            
            # Load merges
            merges = {}
            try:
                merges_file = Path(path) / "merges.txt"
                with open(merges_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split()
                        if len(parts) == 2:
                            merges[(parts[0], parts[1])] = i
            except Exception as e:
                logger.warning(f"Failed to load merges: {e}")
            
            tokenizer.merges = merges
            tokenizer._is_trained = True
            
            # Register special token IDs
            for token_type in ["pad_token", "unk_token", "bos_token", "eos_token"]:
                token = getattr(hf_tokenizer, token_type, None)
                if token:
                    token_id = hf_tokenizer.convert_tokens_to_ids([token])[0]
                    special_tokens.register_special_token_id(token_type, token_id)
            
            logger.info(f"Successfully loaded GPT tokenizer {model_name} from Hugging Face")
            return tokenizer
            
        except ImportError:
            logger.error("transformers and huggingface_hub packages are required to load from Hugging Face")
            raise
        except Exception as e:
            logger.error(f"Failed to load GPT tokenizer from Hugging Face: {e}")
            raise 