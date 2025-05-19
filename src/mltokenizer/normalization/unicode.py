import unicodedata
from typing import Literal

from loguru import logger

UnicodeNormalizationForm = Literal["NFC", "NFKC", "NFD", "NFKD"]

def normalize_unicode(text: str, form: UnicodeNormalizationForm = "NFC") -> str:
    """Applies Unicode normalization to the text."""
    try:
        return unicodedata.normalize(form, text)
    except Exception as e:
        logger.error(f"Unicode normalization failed for form {form} on text (sample: \"{text[:50]}...\"): {e}")
        return text # Return original text on error

class UnicodeNormalizerComponent:
    """A component for applying Unicode normalization as part of a pipeline."""
    def __init__(self, form: UnicodeNormalizationForm = "NFC"):
        self.form = form

    def normalize(self, text: str) -> str:
        return normalize_unicode(text, self.form)

# Add more Unicode related utilities if needed, e.g.:
# - Removing control characters
# - Handling specific script conversions or simplifications (e.g., full-width to half-width)

logger.info("Unicode normalization module loaded.") 