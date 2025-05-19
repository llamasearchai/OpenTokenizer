from typing import Dict, List, Optional, Set, Tuple, Union


class Encoder:
    """Token encoder component for converting between tokens and IDs."""
    
    def __init__(self, token_to_id: Dict[str, int]):
        """Initialize the encoder with a vocabulary.
        
        Args:
            token_to_id: Dictionary mapping tokens to their IDs
        """
        self.token_to_id_map = token_to_id
        self.id_to_token_map = {v: k for k, v in token_to_id.items()}
        self.added_tokens: Set[str] = set()
        self.special_tokens: Dict[str, str] = {}
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.token_to_id_map)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        return self.token_to_id_map.copy()
    
    def token_to_id(self, token: str) -> int:
        """Convert a token to its ID.
        
        Args:
            token: Token to convert
            
        Returns:
            Token ID or 0 if not found (UNK token)
        """
        return self.token_to_id_map.get(token, 0)  # Default to UNK (0)
    
    def id_to_token(self, id: int) -> str:
        """Convert an ID to its token.
        
        Args:
            id: ID to convert
            
        Returns:
            Token string or "[UNK]" if not found
        """
        return self.id_to_token_map.get(id, "[UNK]")
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Encode a list of tokens to IDs.
        
        Args:
            tokens: List of tokens to encode
            
        Returns:
            List of token IDs
        """
        return [self.token_to_id(token) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Decode a list of IDs to tokens.
        
        Args:
            ids: List of IDs to decode
            
        Returns:
            List of tokens
        """
        return [self.id_to_token(id) for id in ids]
    
    def add_tokens(self, tokens: List[str]) -> int:
        """Add new tokens to the vocabulary.
        
        Args:
            tokens: List of tokens to add
            
        Returns:
            Number of tokens actually added
        """
        added = 0
        for token in tokens:
            if token not in self.token_to_id_map:
                new_id = len(self.token_to_id_map)
                self.token_to_id_map[token] = new_id
                self.id_to_token_map[new_id] = token
                self.added_tokens.add(token)
                added += 1
        return added
    
    def add_special_token(self, token: str, token_type: str) -> bool:
        """Add a special token to the vocabulary.
        
        Args:
            token: Special token to add
            token_type: Type of special token (e.g., cls, sep, etc.)
            
        Returns:
            Whether the token was added
        """
        if token not in self.token_to_id_map:
            new_id = len(self.token_to_id_map)
            self.token_to_id_map[token] = new_id
            self.id_to_token_map[new_id] = token
            self.special_tokens[token_type] = token
            return True
        
        # Token already exists, just mark it as special
        self.special_tokens[token_type] = token
        return False
    
    def is_special_token(self, token: str) -> bool:
        """Check if a token is a special token.
        
        Args:
            token: Token to check
            
        Returns:
            Whether the token is a special token
        """
        return token in self.special_tokens.values() 