import re
from typing import List

from loguru import logger

# Placeholder for more advanced cleaning functions or classes

def remove_html_tags(text: str) -> str:
    """Removes HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace (multiple spaces to one, strip leading/trailing)."""
    return " ".join(text.split())

class TextCleaner:
    """A configurable text cleaner."""
    def __init__(self, cleaning_steps: List[str] = None):
        """
        Args:
            cleaning_steps: List of cleaning step names to apply, e.g., ["remove_html", "normalize_whitespace"].
        """
        self.steps = []
        if cleaning_steps:
            for step_name in cleaning_steps:
                if step_name == "remove_html":
                    self.steps.append(remove_html_tags)
                elif step_name == "normalize_whitespace":
                    self.steps.append(normalize_whitespace)
                # Add more predefined steps here
                else:
                    logger.warning(f"Unknown cleaning step: {step_name}")
    
    def clean(self, text: str) -> str:
        """Applies the configured cleaning steps to the text."""
        for step_func in self.steps:
            text = step_func(text)
        return text

# Example predefined pipelines or configurations could go here
DEFAULT_CLEANING_STEPS = ["normalize_whitespace"]
WEB_CONTENT_CLEANING_STEPS = ["remove_html", "normalize_whitespace"] 