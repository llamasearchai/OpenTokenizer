# src/mltokenizer/algorithms/sentencepiece_tokenizer.py
# This will be a wrapper around the sentencepiece library, similar to UnigramTokenizer
# but can also handle SentencePiece's BPE mode.

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json # For saving/loading config
import shutil # For copying files
import os # For os.path.exists, os.remove

from loguru import logger

from mltokenizer.core.base_tokenizer import (
    BaseTokenizer,
    TokenizedOutput,
    TokenizerOptions,
    TokenizerType, # Will use CUSTOM or a new SentencePiece type
)
from mltokenizer.core.errors import TokenizationError, UntrainedTokenizerError
from mltokenizer.encoding.encoder import Encoder
from mltokenizer.encoding.special_tokens import SpecialTokens
from mltokenizer.normalization.normalizers import Normalizer # SP model usually handles normalization
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline


class SentencePieceTokenizer(BaseTokenizer):
    """Wrapper for SentencePiece library (supports Unigram and BPE models)."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None, # Path to a pre-trained SentencePiece model file (.model)
        vocab_size: int = 32000, # Default, but typically determined by the loaded/trained model
        normalizer: Optional[Normalizer] = None, # Usually handled by SP model config
        preprocessor: Optional[PreprocessingPipeline] = None,
        encoder: Optional[Encoder] = None, # Will be built from SP model
        special_tokens: Optional[SpecialTokens] = None,
        options: Optional[TokenizerOptions] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>", # SP uses pad_id, often 0 or -1 (not used)
    ):
        
        st_handler = special_tokens or SpecialTokens(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token
        )

        super().__init__(
            tokenizer_type=TokenizerType.SENTENCEPIECE, # Mark as SentencePiece type
            vocab_size=vocab_size,
            normalizer=normalizer, # SP model often has built-in normalization
            preprocessor=preprocessor or PreprocessingPipeline(),
            encoder=encoder, # Will be populated by load_model
            special_tokens=st_handler,
            options=options or TokenizerOptions(),
        )
        
        self.sp_model = None # SentencePieceProcessor instance
        self._is_trained = False
        self._trained_model_path: Optional[Path] = None # Path to the .model file after training
        self._trained_vocab_path: Optional[Path] = None # Path to the .vocab file after training

        if model_path:
            try:
                self.load_model(model_path)
            except Exception as e:
                logger.error(f"Failed to load SentencePiece model from {model_path} during init: {e}")
        
        logger.debug(f"Initialized SentencePieceTokenizer. Trained: {self.is_trained}, Vocab Size: {self.vocab_size}")

    def _sync_special_tokens_from_sp_model(self):
        if not self.sp_model or not self.special_tokens:
            return
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size())}
        token_types_to_check = ["unk_token", "bos_token", "eos_token", "pad_token"]
        for token_type in token_types_to_check:
            token_str = getattr(self.special_tokens, token_type, None)
            if token_str and token_str in vocab:
                self.special_tokens.register_special_token_id(token_type, vocab[token_str])
            else:
                sp_attr_name = f"{token_type.replace('_token', '')}_id"
                if hasattr(self.sp_model, sp_attr_name):
                    try:
                        token_id_val = getattr(self.sp_model, sp_attr_name)()
                        if token_id_val is not None and token_id_val != -1:
                            self.special_tokens.register_special_token_id(token_type, token_id_val)
                            actual_token_str = self.sp_model.id_to_piece(token_id_val)
                            if token_str != actual_token_str:
                                self.special_tokens.register_special_token(token_type, actual_token_str)
                                logger.info(f"Updated special token string for {token_type} to '{actual_token_str}' from SP model.")
                    except Exception as e:
                        logger.warning(f"Could not retrieve ID for {token_type} from SP model attribute {sp_attr_name}: {e}")
        logger.debug(f"Special tokens after sync with SP model: {self.special_tokens.get_special_tokens_dict()}")
        logger.debug(f"Special token IDs after sync: {self.special_tokens.special_token_ids}")

    def load_model(self, model_path: Union[str, Path]):
        try:
            import sentencepiece as spm
            model_path_str = str(model_path)
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(model_path_str)
            vocab = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size())}
            self.encoder = Encoder(vocab)
            self.vocab_size = self.encoder.vocab_size
            self._is_trained = True
            # Store the path from which this model was loaded, could be a pre-trained one
            self._trained_model_path = Path(model_path_str)
            # Attempt to locate a .vocab file alongside the .model file
            vocab_path_candidate = Path(model_path_str).with_suffix(".vocab")
            if vocab_path_candidate.exists():
                self._trained_vocab_path = vocab_path_candidate
            else:
                self._trained_vocab_path = None # No vocab file found or expected by default

            self._sync_special_tokens_from_sp_model()
            logger.info(f"SentencePiece model loaded from {model_path_str}. Vocab size: {self.vocab_size}")
        except ImportError:
            logger.error("sentencepiece package is required. Please install it: pip install sentencepiece")
            raise
        except Exception as e:
            self._is_trained = False
            logger.error(f"Error loading SentencePiece model from {model_path}: {e}", exc_info=True)
            raise TokenizationError(f"Failed to load SentencePiece model: {e}")

    def train(self, texts: List[str], model_prefix: str = "sp_model", model_type: str = "unigram", vocab_size: Optional[int] = None, **kwargs) -> None:
        if vocab_size is None: vocab_size = self.vocab_size
        
        # Ensure model_prefix is just a name, not a path, for temp file handling
        # The actual files will be created in the current working directory or a temp dir.
        # For simplicity, we'll let SP create them in CWD for now.
        clean_model_prefix = Path(model_prefix).name

        # Define paths for the output model and vocab files
        output_model_file = Path(f"{clean_model_prefix}.model")
        output_vocab_file = Path(f"{clean_model_prefix}.vocab")

        try:
            import sentencepiece as spm
            from tempfile import NamedTemporaryFile
            
            # corpus_file_path should be handled carefully if it's temporary
            corpus_file_path_str = "" # Initialize
            with NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp_file:
                for text_item in texts: tmp_file.write(text_item + "\n")
                corpus_file_path_str = tmp_file.name
            
            # Construct special token arguments for SentencePiece trainer
            # SentencePiece uses control_symbols for tokens that are part of the vocabulary by default (e.g. <s>, </s>)
            # and user_defined_symbols for other custom tokens.
            # Forcing them as control_symbols can ensure they get specific IDs if that's desired over learning them.
            # However, passing them as pieces (unk_piece, bos_piece etc.) is often sufficient.
            sp_train_args = {
                "input": corpus_file_path_str,
                "model_prefix": clean_model_prefix,
                "model_type": model_type,
                "vocab_size": vocab_size,
                "unk_piece": self.special_tokens.unk_token,
                "bos_piece": self.special_tokens.bos_token,
                "eos_piece": self.special_tokens.eos_token,
                "pad_piece": self.special_tokens.pad_token,
                # Add other common SentencePiece arguments if needed or pass via kwargs
                # e.g., character_coverage, max_sentence_length
            }
            
            # Filter out None values for cleaner command string
            sp_train_args = {k: v for k, v in sp_train_args.items() if v is not None}
            
            # Allow overriding or adding more SP arguments via kwargs
            # Example: --character_coverage=0.9995 --model_type=bpe
            additional_sp_args = {}
            if "user_defined_symbols" in kwargs: # Let user explicitly define symbols
                additional_sp_args["user_defined_symbols"] = kwargs.pop("user_defined_symbols")
            if "control_symbols" in kwargs:
                additional_sp_args["control_symbols"] = kwargs.pop("control_symbols")

            # Merge kwargs, giving precedence to explicitly set SP args
            sp_train_args.update(kwargs)
            sp_train_args.update(additional_sp_args) # ensure explicit symbol args are prioritized

            train_cmd_list = []
            for k, v in sp_train_args.items():
                if isinstance(v, list): # for args like user_defined_symbols
                    train_cmd_list.append(f"--{k}={','.join(v)}")
                else:
                    train_cmd_list.append(f"--{k}={v}")
            
            train_cmd_str = " ".join(train_cmd_list)
            logger.info(f"Training SentencePiece model with arguments: {train_cmd_str}")
            
            spm.SentencePieceTrainer.train(train_cmd_str)

            # Store paths to the generated files
            self._trained_model_path = output_model_file.resolve() # Get absolute path
            self._trained_vocab_path = output_vocab_file.resolve()

            if not self._trained_model_path.exists():
                raise TokenizationError(f"SentencePiece training finished but model file '{self._trained_model_path}' not found.")

            self.load_model(self._trained_model_path) # Load the newly trained model
            logger.info(f"SentencePiece training complete. Model: {self._trained_model_path}, Vocab: {self._trained_vocab_path if self._trained_vocab_path.exists() else 'N/A'}")

        except ImportError:
            logger.error("sentencepiece package is required. Please install it: pip install sentencepiece")
            raise
        except Exception as e:
            logger.error(f"Error training SentencePiece model: {e}", exc_info=True)
            raise TokenizationError(f"SentencePiece training failed: {e}")
        finally:
            # Clean up the temporary corpus file
            if corpus_file_path_str and os.path.exists(corpus_file_path_str):
                try:
                    os.remove(corpus_file_path_str)
                    logger.debug(f"Removed temporary corpus file: {corpus_file_path_str}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary corpus file {corpus_file_path_str}: {e}")
            # Clean up model/vocab files if training failed before load_model and they are in CWD
            # However, if they were created by model_prefix, they might be intended output.
            # For now, SP manages its own file creation based on model_prefix.

    def encode(self, text: str, text_pair: Optional[str] = None, add_special_tokens: bool = True, return_tokens: bool = False) -> TokenizedOutput:
        if not self._is_trained or not self.sp_model: raise UntrainedTokenizerError("SentencePieceTokenizer is not trained or model not loaded.")
        normalized_text = self.normalizer.normalize(text) if self.normalizer else text
        processed_text = self.preprocessor.process(normalized_text) if self.preprocessor else normalized_text
        ids1 = self.sp_model.encode_as_ids(processed_text)
        tokens1 = self.sp_model.encode_as_pieces(processed_text) if return_tokens else None
        ids2: Optional[List[int]] = None; tokens2: Optional[List[str]] = None
        if text_pair:
            normalized_pair = self.normalizer.normalize(text_pair) if self.normalizer else text_pair
            processed_pair = self.preprocessor.process(normalized_pair) if self.preprocessor else normalized_pair
            ids2 = self.sp_model.encode_as_ids(processed_pair)
            if return_tokens: tokens2 = self.sp_model.encode_as_pieces(processed_pair)
        final_ids: List[int] = []; final_tt_ids: List[int] = []; final_sp_mask: List[int] = []
        final_tokens_list: Optional[List[str]] = [] if return_tokens else None
        _sp = self.special_tokens; is_pair = ids2 is not None
        if add_special_tokens:
            if is_pair and _sp.cls_token_id is not None and _sp.sep_token_id is not None:
                final_ids.append(_sp.cls_token_id); final_tt_ids.append(0); final_sp_mask.append(1); 
                if return_tokens: final_tokens_list.append(_sp.cls_token)
                final_ids.extend(ids1); final_tt_ids.extend([0]*len(ids1)); final_sp_mask.extend([0]*len(ids1))
                if return_tokens and tokens1: final_tokens_list.extend(tokens1)
                final_ids.append(_sp.sep_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                if return_tokens: final_tokens_list.append(_sp.sep_token)
                final_ids.extend(ids2); final_tt_ids.extend([1]*len(ids2)); final_sp_mask.extend([0]*len(ids2))
                if return_tokens and tokens2: final_tokens_list.extend(tokens2)
                final_ids.append(_sp.sep_token_id); final_tt_ids.append(1); final_sp_mask.append(1)
                if return_tokens: final_tokens_list.append(_sp.sep_token)
            else:
                if _sp.bos_token_id is not None:
                    final_ids.append(_sp.bos_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                    if return_tokens: final_tokens_list.append(_sp.bos_token)
                final_ids.extend(ids1); final_tt_ids.extend([0]*len(ids1)); final_sp_mask.extend([0]*len(ids1))
                if return_tokens and tokens1: final_tokens_list.extend(tokens1)
                if is_pair:
                    if _sp.eos_token_id is not None:
                         final_ids.append(_sp.eos_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                         if return_tokens: final_tokens_list.append(_sp.eos_token)
                    final_ids.extend(ids2); final_tt_ids.extend([0]*len(ids2)); final_sp_mask.extend([0]*len(ids2))
                    if return_tokens and tokens2: final_tokens_list.extend(tokens2)
                if _sp.eos_token_id is not None:
                    final_ids.append(_sp.eos_token_id); final_tt_ids.append(0); final_sp_mask.append(1)
                    if return_tokens: final_tokens_list.append(_sp.eos_token)
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
            pad_token_str = _sp.pad_token or "<pad>"
            final_tokens_list.extend([pad_token_str] * (len(padded_ids) - len(final_tokens_list)))
        if len(final_sp_mask) < len(padded_ids): final_sp_mask.extend([0] * (len(padded_ids) - len(final_sp_mask)))
        return TokenizedOutput(input_ids=padded_ids, attention_mask=attention_mask if self.options.return_attention_mask else None,
                               token_type_ids=padded_tt_ids if self.options.return_token_type_ids and any(id !=0 for id in padded_tt_ids) else None,
                               special_tokens_mask=final_sp_mask if self.options.return_special_tokens_mask else None,
                               overflowing_tokens=overflow if self.options.return_overflowing_tokens else None,
                               length=len(padded_ids) if self.options.return_length else None, tokens=final_tokens_list)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of token IDs back to a string."""
        if not self.sp_model:
            raise UntrainedTokenizerError("SentencePiece model not loaded.")

        ids_to_decode = token_ids
        if skip_special_tokens and self.special_tokens:
            # Get all registered special token IDs
            # self.special_tokens.all_special_ids might not be up-to-date if only strings were set
            # and not IDs, or if SP model defined different IDs. _sync_special_tokens_from_sp_model helps.
            current_special_ids = set(self.special_tokens.special_token_ids.values())
            
            # Also consider default SP model special IDs if not perfectly synced
            # or if they weren't explicitly added to our SpecialTokens handler
            try:
                if self.sp_model.unk_id() != -1: current_special_ids.add(self.sp_model.unk_id())
                if self.sp_model.bos_id() != -1: current_special_ids.add(self.sp_model.bos_id())
                if self.sp_model.eos_id() != -1: current_special_ids.add(self.sp_model.eos_id())
                if self.sp_model.pad_id() != -1: current_special_ids.add(self.sp_model.pad_id())
            except Exception as e:
                logger.warning(f"Could not query all default SP special IDs during decode: {e}")

            if current_special_ids:
                ids_to_decode = [id_val for id_val in token_ids if id_val not in current_special_ids]
            else:
                logger.warning("skip_special_tokens is True, but no special token IDs found in handler to skip.")

        try:
            # SentencePiece expects a list of integers. Some tokenizers might output int32, ensure compatibility.
            decoded_text = self.sp_model.decode([int(id_val) for id_val in ids_to_decode])
            return decoded_text
        except Exception as e:
            logger.error(f"Error during SentencePiece decoding: {e}", exc_info=True)
            # Fallback or re-raise depending on desired robustness
            raise TokenizationError(f"Failed to decode token IDs: {e}")

    def save(self, path: Union[str, Path], model_filename: str = "sp.model", vocab_filename: str = "sp.vocab", **kwargs) -> None:
        """Saves the tokenizer to a directory.

        This involves saving the SentencePiece model file, an optional vocabulary file,
        and a mltokenizer configuration file.
        """
        if not self._is_trained or not self.sp_model:
            raise UntrainedTokenizerError("Cannot save an untrained or model-less SentencePieceTokenizer.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Determine the source model path
        source_model_path = self._trained_model_path
        if not source_model_path or not source_model_path.exists():
            # This case might happen if the tokenizer was loaded from a model_path in __init__
            # and sp_model.serialized_model_proto() is not a file path.
            # Or if train() was not called, but load_model() was.
            # We need a reliable way to get the current model's file path.
            # If self.sp_model.model_file() existed, it would be great. It does not.
            # For now, we rely on self._trained_model_path being set by load_model or train.
            raise TokenizationError(
                "Source SentencePiece model path is not known or does not exist. "
                "Ensure the tokenizer was trained or loaded correctly, and the .model file is accessible."
            )

        # Define target paths for model and vocab within the save_dir
        target_model_file = save_dir / model_filename
        
        # Copy the SentencePiece model file
        try:
            shutil.copyfile(source_model_path, target_model_file)
            logger.info(f"SentencePiece model file copied from {source_model_path} to {target_model_file}")
        except Exception as e:
            logger.error(f"Failed to copy SentencePiece model file from {source_model_path} to {target_model_file}: {e}")
            raise TokenizationError(f"Failed to save SentencePiece model file: {e}")

        # Handle vocabulary file (if it exists and is tracked)
        target_vocab_file_entry = None
        if self._trained_vocab_path and self._trained_vocab_path.exists():
            target_vocab_file = save_dir / vocab_filename
            try:
                shutil.copyfile(self._trained_vocab_path, target_vocab_file)
                logger.info(f"SentencePiece vocab file copied from {self._trained_vocab_path} to {target_vocab_file}")
                target_vocab_file_entry = vocab_filename
            except Exception as e:
                logger.warning(f"Failed to copy SentencePiece vocab file from {self._trained_vocab_path} to {target_vocab_file}: {e}. Vocab file will not be part of the saved artifact.")
        elif Path(str(source_model_path).replace(".model", ".vocab")).exists(): # Check common naming convention
            # If _trained_vocab_path wasn't set but a .vocab file exists next to .model
            source_vocab_path_candidate = Path(str(source_model_path).replace(".model", ".vocab"))
            target_vocab_file = save_dir / vocab_filename
            try:
                shutil.copyfile(source_vocab_path_candidate, target_vocab_file)
                logger.info(f"Found and copied SentencePiece vocab file {source_vocab_path_candidate} to {target_vocab_file}")
                target_vocab_file_entry = vocab_filename
            except Exception as e:
                logger.warning(f"Failed to copy SentencePiece vocab file from {source_vocab_path_candidate} to {target_vocab_file}: {e}. Vocab file will not be part of the saved artifact.")


        config = {
            "tokenizer_type": self.tokenizer_type.value,
            "model_file": model_filename, # Relative path within the save_dir
            "vocab_file": target_vocab_file_entry, # Relative path, or None
            "vocab_size": self.vocab_size,
            "special_tokens_map": self.special_tokens.get_special_tokens_dict(),
            "special_token_ids": self.special_tokens.special_token_ids,
            "options": self.options.dict(),
            # Placeholders for normalizer and preprocessor configurations
            "normalizer_config": self.normalizer.config if self.normalizer else None,
            "preprocessor_config": self.preprocessor.config if self.preprocessor else None,
            # Add any other relevant SentencePiece model parameters if needed
            # e.g., model_type if we want to store it explicitly outside of SP model
        }

        config_file_path = save_dir / "mltokenizer_config.json"
        try:
            with open(config_file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Tokenizer configuration saved to {config_file_path}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer configuration to {config_file_path}: {e}")
            raise TokenizationError(f"Failed to save tokenizer configuration: {e}")

        logger.info(f"SentencePieceTokenizer saved to {save_dir}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "SentencePieceTokenizer":
        """Loads the tokenizer from a directory."""
        load_dir = Path(path)
        config_file_path = load_dir / "mltokenizer_config.json"

        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file 'mltokenizer_config.json' not found in {load_dir}")

        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load or parse configuration from {config_file_path}: {e}")
            raise TokenizationError(f"Failed to load tokenizer configuration: {e}")

        model_file_name = config.get("model_file")
        if not model_file_name:
            raise TokenizationError("Model file not specified in configuration.")
        
        actual_model_path = load_dir / model_file_name
        if not actual_model_path.exists():
            raise FileNotFoundError(f"SentencePiece model file '{actual_model_path}' not found in {load_dir}")

        # Reconstruct SpecialTokens
        special_tokens_map = config.get("special_tokens_map", {})
        special_token_ids = config.get("special_token_ids", {}) # These might be re-synced after load
        
        # Ensure all standard token types are present in special_tokens_map even if None
        # for consistent SpecialTokens object creation.
        for token_key in ["unk_token", "bos_token", "eos_token", "pad_token", "cls_token", "sep_token", "mask_token"]:
            if token_key not in special_tokens_map:
                special_tokens_map[token_key] = None # Explicitly set to None if missing

        special_tokens_handler = SpecialTokens(**special_tokens_map)
        # Restore known IDs, _sync_special_tokens_from_sp_model will verify/update them
        for name, token_id in special_token_ids.items():
            if token_id is not None: # Only register if ID is valid
                 # The name in special_token_ids should directly correspond to an attribute in SpecialTokens
                 # e.g., "unk_token_id" maps to "unk_token".
                 # Or, it's a custom token name. For now, assume it maps to standard ones.
                 # This part might need refinement if custom tokens are stored with their IDs here.
                 # For now, we rely on _sync_special_tokens_from_sp_model.
                 pass


        # Reconstruct TokenizerOptions
        options_dict = config.get("options", {})
        tokenizer_options = TokenizerOptions(**options_dict)
        
        # Vocab size from config
        vocab_size = config.get("vocab_size", 32000) # Default if not in config

        # TODO: Reconstruct Normalizer and Preprocessor from their configs
        # normalizer_config = config.get("normalizer_config")
        # preprocessor_config = config.get("preprocessor_config")
        # For now, they will be None or default.

        # Create tokenizer instance without model_path initially, then load
        # Pass through any additional kwargs the user might want for __init__
        # but prioritize loaded config values.
        
        # Kwargs for __init__ should not override these core components from config
        init_kwargs = {
            "vocab_size": vocab_size,
            "special_tokens": special_tokens_handler,
            "options": tokenizer_options,
            # "normalizer": reconstructed_normalizer,
            # "preprocessor": reconstructed_preprocessor,
        }
        init_kwargs.update(kwargs) # User can override defaults if not in config, or add others

        tokenizer = cls(**init_kwargs)
        
        # Now load the actual SentencePiece model
        tokenizer.load_model(actual_model_path)
        
        # _sync_special_tokens_from_sp_model is called within load_model,
        # which should update special_token_ids in special_tokens_handler
        # based on the loaded SP model.

        # If vocab_file was saved, store its path (mainly for reference, SP model is self-contained)
        vocab_file_name = config.get("vocab_file")
        if vocab_file_name:
            actual_vocab_path = load_dir / vocab_file_name
            if actual_vocab_path.exists():
                tokenizer._trained_vocab_path = actual_vocab_path
            else:
                logger.warning(f"Vocab file '{vocab_file_name}' specified in config but not found at {actual_vocab_path}")
        
        logger.info(f"SentencePieceTokenizer loaded from {load_dir}. Vocab size: {tokenizer.vocab_size}")
        return tokenizer

    @property
    def trained_model_path(self):
        return self._trained_model_path

    @property
    def trained_vocab_path(self):
        return self._trained_vocab_path

__all__ = ["SentencePieceTokenizer"] 