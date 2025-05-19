from typing import Callable, List, Optional, Union

from tokenizers.preprocessing.text_cleaning import TextCleaner
from tokenizers.utils.logging import get_module_logger


logger = get_module_logger("preprocessing")


class PreprocessingStep:
    """A single preprocessing step."""
    
    def __init__(self, name: str, func: Callable[[str], str]):
        """Initialize a preprocessing step.
        
        Args:
            name: Name of the step
            func: Function that takes a string and returns a processed string
        """
        self.name = name
        self.func = func
    
    def __call__(self, text: str) -> str:
        """Apply the preprocessing step.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text
        """
        return self.func(text)


class PreprocessingPipeline:
    """Pipeline for preprocessing text."""
    
    def __init__(self, steps: Optional[List[PreprocessingStep]] = None):
        """Initialize a preprocessing pipeline.
        
        Args:
            steps: List of preprocessing steps
        """
        self.steps = steps or []
        self.logger = logger
    
    def add_step(self, step: Union[PreprocessingStep, Callable[[str], str]], name: Optional[str] = None) -> None:
        """Add a preprocessing step to the pipeline.
        
        Args:
            step: Preprocessing step or function
            name: Name for the step (required if step is a function)
        """
        if isinstance(step, PreprocessingStep):
            self.steps.append(step)
        elif callable(step):
            if name is None:
                name = getattr(step, "__name__", f"step_{len(self.steps)}")
            self.steps.append(PreprocessingStep(name, step))
        else:
            raise TypeError("Step must be a PreprocessingStep or a callable")
    
    def remove_step(self, name: str) -> bool:
        """Remove a preprocessing step by name.
        
        Args:
            name: Name of the step to remove
            
        Returns:
            Whether the step was successfully removed
        """
        for i, step in enumerate(self.steps):
            if step.name == name:
                self.steps.pop(i)
                return True
        return False
    
    def process(self, text: str) -> str:
        """Apply all preprocessing steps to text.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text
        """
        for step in self.steps:
            try:
                text = step(text)
            except Exception as e:
                self.logger.error(f"Error in preprocessing step '{step.name}': {e}")
                # Continue with other steps
        
        return text
    
    def __len__(self) -> int:
        """Get the number of steps in the pipeline."""
        return len(self.steps)