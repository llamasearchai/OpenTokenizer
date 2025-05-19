from typing import List, Optional, Tuple, Union

from tokenizers.utils.logging import get_module_logger


logger = get_module_logger("sequence_handling")


class SequenceHandler:
    """Handler for sequence preprocessing operations."""
    
    @staticmethod
    def truncate_sequence(
        tokens: List[str],
        max_length: int,
        truncation_strategy: str = "longest_first",
        stride: int = 0
    ) -> Tuple[List[str], Optional[List[List[str]]]]:
        """Truncate a sequence to a maximum length.
        
        Args:
            tokens: List of tokens to truncate
            max_length: Maximum length
            truncation_strategy: Truncation strategy 
            stride: Stride for truncation with overlap
            
        Returns:
            Tuple of (truncated tokens, overflowing tokens)
        """
        if len(tokens) <= max_length:
            return tokens, None
        
        # Simple truncation
        truncated = tokens[:max_length]
        
        # Handle overflowing tokens with stride
        overflowing_tokens = None
        if stride > 0:
            overflowing_tokens = []
            for i in range(0, len(tokens) - max_length + 1, stride):
                if i > 0:  # Skip the first one as it's the main output
                    overflowing_tokens.append(tokens[i:i + max_length])
        
        return truncated, overflowing_tokens
    
    @staticmethod
    def truncate_sequences(
        first_tokens: List[str],
        second_tokens: Optional[List[str]] = None,
        max_length: int = 512,
        truncation_strategy: str = "longest_first",
        stride: int = 0
    ) -> Tuple[List[str], Optional[List[str]], Optional[List[List[str]]]]:
        """Truncate two sequences to fit within max_length.
        
        Args:
            first_tokens: First sequence of tokens
            second_tokens: Second sequence of tokens (optional)
            max_length: Maximum combined length
            truncation_strategy: Strategy for truncation
            stride: Stride for truncation with overlap
            
        Returns:
            Tuple of (truncated first tokens, truncated second tokens, overflowing tokens)
        """
        if second_tokens is None:
            truncated, overflowing = SequenceHandler.truncate_sequence(
                first_tokens, max_length, truncation_strategy, stride
            )
            return truncated, None, overflowing
        
        total_len = len(first_tokens) + len(second_tokens)
        if total_len <= max_length:
            return first_tokens, second_tokens, None
        
        # Apply truncation strategy
        if truncation_strategy == "longest_first":
            # Truncate the longer sequence first
            if len(first_tokens) > len(second_tokens):
                first_truncated_len = max_length - len(second_tokens)
                first_truncated = first_tokens[:first_truncated_len]
                return first_truncated, second_tokens, None
            else:
                second_truncated_len = max_length - len(first_tokens)
                second_truncated = second_tokens[:second_truncated_len]
                return first_tokens, second_truncated, None
        elif truncation_strategy == "only_first":
            # Only truncate the first sequence
            first_truncated_len = max_length - len(second_tokens)
            if first_truncated_len <= 0:
                logger.warning("Second sequence is longer than max_length, truncating it instead")
                return [], second_tokens[:max_length], None
            first_truncated = first_tokens[:first_truncated_len]
            return first_truncated, second_tokens, None
        elif truncation_strategy == "only_second":
            # Only truncate the second sequence
            second_truncated_len = max_length - len(first_tokens)
            if second_truncated_len <= 0:
                logger.warning("First sequence is longer than max_length, truncating it instead")
                return first_tokens[:max_length], [], None
            second_truncated = second_tokens[:second_truncated_len]
            return first_tokens, second_truncated, None
        else:
            # Default to truncating both proportionally
            ratio = len(first_tokens) / total_len
            first_truncated_len = int(max_length * ratio)
            second_truncated_len = max_length - first_truncated_len
            return first_tokens[:first_truncated_len], second_tokens[:second_truncated_len], None
    
    @staticmethod
    def pad_sequence(
        tokens: List[str],
        max_length: int,
        pad_token: str = "[PAD]",
        padding_side: str = "right"
    ) -> Tuple[List[str], List[int]]:
        """Pad a sequence to a specified length.
        
        Args:
            tokens: List of tokens to pad
            max_length: Length to pad to
            pad_token: Token to use for padding
            padding_side: Side to add padding ("left" or "right")
            
        Returns:
            Tuple of (padded tokens, attention mask)
        """
        padding_length = max_length - len(tokens)
        if padding_length <= 0:
            return tokens, [1] * len(tokens)
        
        if padding_side == "right":
            padded_tokens = tokens + [pad_token] * padding_length
            attention_mask = [1] * len(tokens) + [0] * padding_length
        else:
            padded_tokens = [pad_token] * padding_length + tokens
            attention_mask = [0] * padding_length + [1] * len(tokens)
        
        return padded_tokens, attention_mask