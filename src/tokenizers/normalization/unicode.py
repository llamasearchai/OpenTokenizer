import unicodedata
from typing import Optional

from tokenizers.normalization.normalizers import Normalizer


class UnicodeNormalizer(Normalizer):
    """Normalizer that applies Unicode normalization."""
    
    def __init__(self, form: str = "NFKC"):
        """Initialize Unicode normalizer.
        
        Args:
            form: Normalization form (NFC, NFKC, NFD, NFKD)
        """
        self.form = form
        
        # Validate form
        valid_forms = ["NFC", "NFKC", "NFD", "NFKD"]
        if form not in valid_forms:
            raise ValueError(f"Invalid normalization form '{form}'. Use one of: {valid_forms}")
    
    def normalize(self, text: str) -> str:
        """Apply Unicode normalization to text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize(self.form, text)