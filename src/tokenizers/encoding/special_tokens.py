from typing import Dict, List, Optional, Set


class SpecialTokens:
    """Handler for special tokens used in tokenization."""
    
    def __init__(
        self,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        cls_token: Optional[str] = "[CLS]",
        sep_token: Optional[str] = "[SEP]",
        mask_token: Optional[str] = "[MASK]",
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        additional_special_tokens: Optional[List[str]] = None,
    ):
        """Initialize special tokens handler.
        
        Args:
            pad_token: Padding token
            unk_token: Unknown token
            cls_token: Classification token (for BERT-like models)
            sep_token: Separator token (for BERT-like models)
            mask_token: Mask token (for BERT-like models)
            bos_token: Beginning of sequence token (for GPT-like models)
            eos_token: End of sequence token (for GPT-like models)
            additional_special_tokens: Other special tokens
        """
        self.special_tokens = {}
        self.special_token_ids = {}
        
        # Set default tokens
        self.register_special_token("pad_token", pad_token)
        self.register_special_token("unk_token", unk_token)
        
        # Set optional tokens
        if cls_token:
            self.register_special_token("cls_token", cls_token)
        if sep_token:
            self.register_special_token("sep_token", sep_token)
        if mask_token:
            self.register_special_token("mask_token", mask_token)
        if bos_token:
            self.register_special_token("bos_token", bos_token)
        if eos_token:
            self.register_special_token("eos_token", eos_token)
        
        # Additional special tokens
        self.additional_special_tokens = set(additional_special_tokens or [])
    
    def register_special_token(self, token_type: str, token: str) -> None:
        """Register a special token.
        
        Args:
            token_type: Type of token (e.g., pad_token)
            token: The token string
        """
        self.special_tokens[token_type] = token
    
    def register_special_token_id(self, token_type: str, token_id: int) -> None:
        """Register a special token ID.
        
        Args:
            token_type: Type of token (e.g., pad_token)
            token_id: The token ID
        """
        self.special_token_ids[token_type] = token_id
    
    def register_special_tokens(self, special_tokens_dict: Dict[str, str]) -> None:
        """Register multiple special tokens from a dictionary.
        
        Args:
            special_tokens_dict: Dictionary mapping token types to tokens
        """
        for token_type, token in special_tokens_dict.items():
            self.register_special_token(token_type, token)
    
    def get_special_tokens_dict(self) -> Dict[str, str]:
        """Get the dictionary of special tokens.
        
        Returns:
            Dictionary mapping token types to tokens
        """
        return self.special_tokens.copy()
    
    @property
    def pad_token(self) -> Optional[str]:
        """Get the padding token."""
        return self.special_tokens.get("pad_token")
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get the padding token ID."""
        return self.special_token_ids.get("pad_token")
    
    @property
    def unk_token(self) -> Optional[str]:
        """Get the unknown token."""
        return self.special_tokens.get("unk_token")
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """Get the unknown token ID."""
        return self.special_token_ids.get("unk_token")
    
    @property
    def cls_token(self) -> Optional[str]:
        """Get the classification token."""
        return self.special_tokens.get("cls_token")
    
    @property
    def cls_token_id(self) -> Optional[int]:
        """Get the classification token ID."""
        return self.special_token_ids.get("cls_token")
    
    @property
    def sep_token(self) -> Optional[str]:
        """Get the separator token."""
        return self.special_tokens.get("sep_token")
    
    @property
    def sep_token_id(self) -> Optional[int]:
        """Get the separator token ID."""
        return self.special_token_ids.get("sep_token")
    
    @property
    def mask_token(self) -> Optional[str]:
        """Get the mask token."""
        return self.special_tokens.get("mask_token")
    
    @property
    def mask_token_id(self) -> Optional[int]:
        """Get the mask token ID."""
        return self.special_token_ids.get("mask_token")
    
    @property
    def bos_token(self) -> Optional[str]:
        """Get the beginning of sequence token."""
        return self.special_tokens.get("bos_token")
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Get the beginning of sequence token ID."""
        return self.special_token_ids.get("bos_token")
    
    @property
    def eos_token(self) -> Optional[str]:
        """Get the end of sequence token."""
        return self.special_tokens.get("eos_token")
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get the end of sequence token ID."""
        return self.special_token_ids.get("eos_token")
    
    @property
    def all_special_tokens(self) -> List[str]:
        """Get all special tokens.
        
        Returns:
            List of all special tokens
        """
        tokens = list(self.special_tokens.values())
        tokens.extend(self.additional_special_tokens)
        return tokens
    
    @property
    def all_special_ids(self) -> List[int]:
        """Get all special token IDs.
        
        Returns:
            List of all special token IDs
        """
        return list(self.special_token_ids.values())