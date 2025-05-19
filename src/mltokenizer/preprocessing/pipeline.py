from typing import List, Callable

class PreprocessingStep:
    # Placeholder for a generic preprocessing step, can be expanded
    def __init__(self, processor_func: Callable[[str], str]):
        self.processor_func = processor_func

    def process(self, text: str) -> str:
        return self.processor_func(text)

class PreprocessingPipeline:
    """A pipeline for applying multiple preprocessing steps."""
    def __init__(self, steps: List[Callable[[str], str]] = None):
        # steps are functions that take a string and return a string
        self.steps = steps if steps is not None else []

    def add_step(self, step_func: Callable[[str], str]):
        self.steps.append(step_func)

    def process(self, text: str) -> str:
        """Process text through all registered steps."""
        for step_func in self.steps:
            text = step_func(text)
        return text

    # Convenience methods for common preprocessing tasks can be added here
    # For example:
    # @staticmethod
    # def to_lowercase(text: str) -> str:
    #     return text.lower()

__all__ = ["PreprocessingPipeline", "PreprocessingStep"] 