from typing import Dict, Callable

from loguru import logger

# This module is intended for language-specific normalization rules.
# For example, handling German umlauts, French ligatures if not covered by Unicode norm,
# or language-specific character replacements.

# Type for a language-specific normalization function
LangNormFunc = Callable[[str], str]

# Registry for language-specific normalizers
LANGUAGE_NORMALIZERS: Dict[str, LangNormFunc] = {}

def register_lang_normalizer(lang_code: str, normalizer_func: LangNormFunc):
    """Registers a normalizer function for a given language code (e.g., 'de', 'fr')."""
    if lang_code in LANGUAGE_NORMALIZERS:
        logger.warning(f"Overwriting normalizer for language: {lang_code}")
    LANGUAGE_NORMALIZERS[lang_code] = normalizer_func
    logger.info(f"Registered normalizer for language: {lang_code}")

def get_lang_normalizer(lang_code: str) -> Optional[LangNormFunc]:
    """Retrieves a normalizer for a given language code."""
    return LANGUAGE_NORMALIZERS.get(lang_code)

class LanguageSpecificNormalizer:
    """Applies a registered language-specific normalizer."""
    def __init__(self, lang_code: str):
        self.lang_code = lang_code
        self.normalizer_func = get_lang_normalizer(lang_code)
        if not self.normalizer_func:
            logger.warning(f"No normalizer found for language: {self.lang_code}. This component will be a no-op.")

    def normalize(self, text: str) -> str:
        if self.normalizer_func:
            return self.normalizer_func(text)
        return text

# --- Example Language Specific Normalizers ---

def normalize_german_text(text: str) -> str:
    """Example: Specific normalization for German text (e.g., old spelling rules)."""
    # Replace ß with ss after certain vowels if a specific rule is desired beyond Unicode
    # text = text.replace("iß", "iss") # Highly contextual, be careful
    # This is just a placeholder for more complex rules.
    logger.debug(f"Applying German-specific normalization (placeholder) to: {text[:30]}...")
    return text

# register_lang_normalizer("de", normalize_german_text)


logger.info("Language-specific normalization module loaded. Register custom language normalizers here.") 