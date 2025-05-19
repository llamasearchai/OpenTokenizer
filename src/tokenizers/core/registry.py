from typing import Dict, List, Optional, Type

from loguru import logger

from tokenizers.core.base_tokenizer import BaseTokenizer
from tokenizers.utils.logging import registry_logger


class TokenizerRegistry:
    """Registry for tokenizers in the system."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(TokenizerRegistry, cls).__new__(cls)
            cls._instance._tokenizers = {}
            cls._instance.logger = registry_logger

            cls._instance.logger.info("Initialized tokenizer registry")
        return cls._instance
    
    def register_tokenizer(self, tokenizer_id: str, tokenizer: BaseTokenizer) -> None:
        """Register a tokenizer with the registry.
        
        Args:
            tokenizer_id: Unique identifier for the tokenizer
            tokenizer: Tokenizer instance to register
        """
        if tokenizer_id in self._tokenizers:
            self.logger.warning(f"Tokenizer '{tokenizer_id}' already registered. Overwriting.")
        
        self._tokenizers[tokenizer_id] = tokenizer
        self.logger.info(f"Registered tokenizer '{tokenizer_id}' of type {tokenizer.tokenizer_type.value}")
    
    def get_tokenizer(self, tokenizer_id: str) -> Optional[BaseTokenizer]:
        """Get a tokenizer by ID.
        
        Args:
            tokenizer_id: ID of the tokenizer to retrieve
            
        Returns:
            Tokenizer instance or None if not found
        """
        tokenizer = self._tokenizers.get(tokenizer_id)
        if tokenizer is None:
            self.logger.warning(f"Tokenizer '{tokenizer_id}' not found in registry")
        
        return tokenizer
    
    def list_tokenizers(self) -> Dict[str, BaseTokenizer]:
        """List all registered tokenizers.
        
        Returns:
            Dictionary of tokenizer IDs to tokenizer instances
        """
        return self._tokenizers.copy()
    
    def remove_tokenizer(self, tokenizer_id: str) -> bool:
        """Remove a tokenizer from the registry.
        
        Args:
            tokenizer_id: ID of the tokenizer to remove
            
        Returns:
            Whether the tokenizer was removed
        """
        if tokenizer_id in self._tokenizers:
            del self._tokenizers[tokenizer_id]
            self.logger.info(f"Removed tokenizer '{tokenizer_id}' from registry")
            return True
        
        self.logger.warning(f"Tokenizer '{tokenizer_id}' not found in registry")
        return False
    
    def clear(self) -> None:
        """Clear all tokenizers from the registry."""
        self._tokenizers.clear()
        self.logger.info("Cleared tokenizer registry")