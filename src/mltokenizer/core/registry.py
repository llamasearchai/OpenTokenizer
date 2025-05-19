from typing import Dict, Optional
from mltokenizer.core.base_tokenizer import BaseTokenizer
from loguru import logger

class TokenizerRegistry:
    def __init__(self):
        self._tokenizers: Dict[str, BaseTokenizer] = {}
        logger.info("TokenizerRegistry initialized.")

    def register_tokenizer(self, tokenizer_id: str, tokenizer: BaseTokenizer):
        if tokenizer_id in self._tokenizers:
            logger.warning(f"Tokenizer ID '{tokenizer_id}' is being overwritten.")
        self._tokenizers[tokenizer_id] = tokenizer
        logger.info(f"Tokenizer '{tokenizer_id}' registered.")

    def get_tokenizer(self, tokenizer_id: str) -> Optional[BaseTokenizer]:
        tokenizer = self._tokenizers.get(tokenizer_id)
        if not tokenizer:
            logger.warning(f"Tokenizer ID '{tokenizer_id}' not found in registry.")
        return tokenizer

    def list_tokenizers(self) -> Dict[str, BaseTokenizer]:
        return self._tokenizers.copy() 