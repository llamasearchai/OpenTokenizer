import re
from typing import List, Optional, Pattern

from tokenizers.utils.logging import get_module_logger


logger = get_module_logger("text_cleaning")


class TextCleaner:
    """Utilities for cleaning text."""
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return text.lower()
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove numbers from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return re.sub(r'\d+', '', text)
    
    @staticmethod
    def replace_patterns(text: str, patterns: List[tuple]) -> str:
        """Replace patterns in text.
        
        Args:
            text: Text to clean
            patterns: List of (pattern, replacement) tuples
            
        Returns:
            Cleaned text
        """
        for pattern, replacement in patterns:
            if isinstance(pattern, str):
                pattern = re.compile(pattern)
            text = pattern.sub(replacement, text)
        return text
    
    @staticmethod
    def create_standard_pipeline() -> List[tuple]:
        """Create a standard pipeline of text cleaning functions.
        
        Returns:
            List of (name, function) tuples for standard cleaning
        """
        return [
            ("remove_urls", TextCleaner.remove_urls),
            ("remove_html_tags", TextCleaner.remove_html_tags),
            ("remove_extra_whitespace", TextCleaner.remove_extra_whitespace)
        ]
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return re.sub(r'\s+', ' ', text.strip())
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return re.sub(r'<.*?>', '', text)
    
    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0000257F"  # Enclosed characters
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return re.sub(r'[^\w\\s]', '', text)